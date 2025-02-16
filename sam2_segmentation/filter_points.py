import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
import os
import glob
import open3d as o3d
model_path = 'yolo_model.pt'
yolo_model = YOLO(model_path)
# sam2 setup        
checkpoint =  "sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Create a pipeline for bag playback
import pyrealsense2 as rs

import torch
import threading






def segment_pointcloud(frame, points):
    # YOLO detection and SAM2 segmentation (using existing code)
    results = yolo_model.predict(source=frame, conf=0.5, verbose=True)
    if len(results[0].boxes) == 0:
        print("No object detected")
        return None
    else:
        detection_results = {}
        masks_list = []
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy().astype(float)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(frame)
            for i, box in enumerate(bboxes):
                input_box = np.array(box).reshape(1, 4)
                masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
                mask = (masks > 0).astype(np.uint8) * 255
                masks_list.append(mask[0])

        # Create a composite visualization (optional; not used in output)
        visualization_frame = frame.copy()
        for i, mask in enumerate(masks_list):
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = [0, 255, 0]  # Green overlay
            visualization_frame = cv2.addWeighted(visualization_frame, 1, colored_mask, 0.5, 0)
            # Annotate with index (optional)
            coords = np.column_stack(np.where(mask > 0))
            if coords.size > 0:
                y, x = coords.mean(axis=0).astype(int)
                cv2.putText(visualization_frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite("segmented_visualization.png", visualization_frame)
        print("Segmented image saved as segmented_visualization.png")
        print(f"Number of detections: {len(masks_list)}")

        # Prepare output: each detection has filtered points and its own segmented image
        detection_results = {}
        # Reshape points to align with the image dimensions (assumes image shape is 480 x 848)
        points_reshaped = points.reshape(480, 848, 3)
        for idx, mask in enumerate(masks_list):
            # Extract point cloud for detection: only points where the mask is active
            detection_points = points_reshaped[mask > 0]
            # Create a segmented image: zero out areas not in the mask
            detection_image = np.zeros_like(frame)
            detection_image[mask > 0] = frame[mask > 0]
            detection_results[f"detection_{idx}"] = {"points": detection_points, "image": detection_image}

        return detection_results

def extra_filter(points):
    if points.shape[0] > 0:
        # Compute centroid and distances to it
        
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        # Use median absolute deviation to set a robust threshold
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
            # Set threshold as median + 3 * MAD (avoid zero MAD)
        threshold = median_dist + 3 * (mad if mad > 0 else 1e-6)
            # Keep only inliers that are not too far from the centroid
        inliers = points[distances <= threshold]
        print(f"number of inliers {len(inliers)}")
        if len(inliers) > 0:
            filtered_points = inliers.copy()
            return filtered_points
    return None


#20250215_160838.bag, 20250215_165825.bag, /home/ali/Documents/20250215_165438.bag
bag_file = 'bags/20250216_152743.bag'

# Initialize the pipeline and configure it to read from a .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file)

pipeline.start(config)

# Get the depth sensor's depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Frame counter for saving images
# Instead, create and visualize a point cloud every 10 frames using the RealSense point cloud object.
pc = rs.pointcloud()  # RealSense point cloud object

frame_counter = 0
while True:
    try:
        # Wait for frames (timeout to catch end of bag file)
        frames = pipeline.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        print("End of bag file reached.")
        break

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("No frames found, finishing frame extraction.")
        break

    frame_counter += 1

    if frame_counter % 10 == 0:
        # ----- Depth Image Filtering (Median Blur) -----
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_filtered = cv2.medianBlur(depth_image, 5)

        # ----- Point Cloud Generation -----
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        pts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # # Instead of filtering out points, set points outside threshold to [0,0,0]
        valid_mask = pts[:, 2] < 2.0
        pts = np.where(valid_mask[:, None], pts, np.zeros_like(pts))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # # # ----- Voxel Downsampling with Replacement -----
        voxel_size = 0.05
        pts_array = np.asarray(pcd.points)
        voxel_indices = np.floor(pts_array / voxel_size).astype(np.int32)
        _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
        min_points_per_voxel = 30
        voxel_mask = counts[inverse] >= min_points_per_voxel
        pts_array = np.where(voxel_mask[:, None], pts_array, np.zeros_like(pts_array))
        pcd.points = o3d.utility.Vector3dVector(pts_array)

        # ----- Bounding Box Filter with Replacement -----
        pts_np = np.asarray(pcd.points)
        bbox_mask = np.all((pts_np >= -1.0) & (pts_np <= 1.0), axis=1)
        pts_np = np.where(bbox_mask[:, None], pts_np, np.zeros_like(pts_np))

        pcd.points = o3d.utility.Vector3dVector(pts_np)
        


        
        #o3d.visualization.draw_geometries([pcd])

        # # ----- Visualize the RGB Image with Correct Color Channels -----
        rgb_image = np.asanyarray(color_frame.get_data())
        # Convert from RGB to BGR for proper display with OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        segmented_points = segment_pointcloud(np.array(bgr_image),  pts_np) # should have the same shape as original pcd
        if segmented_points is None:
            print("No segmented points found.")
            continue


        
        
        def show_pointcloud(points, window_title):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_title)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()

        print(f"segmented_points keys {segmented_points.keys()}")

        for detection, data in segmented_points.items():
            filtered_points = extra_filter(data["points"])
            image = data["image"]
            if filtered_points is not None:
                threading.Thread(target=show_pointcloud, args=(filtered_points, detection), daemon=True).start()
                
                # cv2.imshow(detection, data["image"])
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
     
       
        # cv2.imshow("RGB Image", bgr_image)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        

print("Finished processing bag file.")
pipeline.stop()
cv2.destroyAllWindows()
