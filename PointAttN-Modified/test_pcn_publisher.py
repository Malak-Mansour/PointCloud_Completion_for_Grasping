import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.train_utils import *
from dataset import PCN_pcd
import h5py
from PCDDataset import PCDDataset
import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs_py.point_cloud2 import read_points, create_cloud

from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
#from custom_srvs.srv import Detection3dArray as ServiceDetection3dArray

import open3d as o3d
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO, FastSAM
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import PointField

# model_cfg = "sam2_hiera_s.yaml"
        # self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

import tf_transformations
from std_srvs.srv import Empty as Empty_srv  # Ensure Empty is imported
from std_msgs.msg import Empty
import time
from std_srvs.srv import Trigger
import glob
import torch
import open3d as o3d
import numpy as np


def rgb_callback(data):
    try:
        # Convert Image message to numpy array directly
        img_array = np.array(data.data, dtype=np.uint8).reshape((data.height, data.width, 3))
        # Convert from RGB to BGR for OpenCV
        node.rgb_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        node.last_timestamp = data.header.stamp
        
        
    except Exception as e:
        node.get_logger().error("Error converting RGB image: %s" % str(e))

def pointcloud_callback(msg):
    try:
        node.pointcloud = msg
    except Exception as e:
        node.get_logger().error("Error converting pointcloud: %s" % str(e))
    




def get_segmentation_masks():
    results = node.yolo_model.predict(source=node.rgb_image, conf=0.5, verbose=True)
    if len(results) == 0:
        node.get_logger().warn("No objects detected")
        #node.reset_pub.publish(Empty())
        return []
    
    # Get detection results from YOLO
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
   
    # Create a copy for overlaying segmentation masks
    segmented_image = node.rgb_image.copy()
    masks_list = []
    
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        node.predictor.set_image(node.rgb_image)
        for box in bboxes:
            input_box = np.array(box).reshape(1, 4)
            # Predict the segmentation mask for the given bounding box
            masks, _, _ = node.predictor.predict(box=input_box, multimask_output=False)
            # Convert the mask to binary (0 and 255)
            mask = (masks > 0).astype(np.uint8) * 255
            mask_bin = mask[0]
            
            # Discard the mask if the segmented points exceed 1500
            print(cv2.countNonZero(mask_bin))
            if cv2.countNonZero(mask_bin) > 3000:
                continue
            
            masks_list.append(mask_bin)
            # Create a colored overlay for the mask (using red)
            colored_mask = np.zeros_like(segmented_image)
            colored_mask[:, :, 2] = mask_bin  # Red channel
            alpha = 0.5  # blending factor
            segmented_image = cv2.addWeighted(segmented_image, 1, colored_mask, alpha, 0)
    
    print(len(masks_list))
    return masks_list


def process_filtered_points(filtered_points):
    all_points = []
    all_keys = []
    for key, pts in filtered_points.items():
        if pts.size == 0:
            continue
        # Create an Open3D point cloud from the filtered points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Downsample or pad to have exactly 512 points
        if len(pts) >= 512:
            downsampled = pcd.farthest_point_down_sample(512)
            pts_processed = np.asarray(downsampled.points)
        else:
            original = np.asarray(pcd.points)
            repeats = int(np.ceil(512 / len(original)))
            pts_processed = np.tile(original, (repeats, 1))[:512]
        all_points.append(pts_processed)
        all_keys.append(key)
    return all_points, all_keys


def save_data(filtered_points, transforms):

    node.reset_pub.publish(Empty())
    # Iterate over each group of filtered points
    for key, pts in filtered_points.items():
        if pts.shape[0] > 200:
            # Calculate the centroid using double precision
            # centroid = np.mean(pts, axis=0, dtype=np.float64)
            # print(centroid)
            # # Compute Euclidean distances with high precision (double precision)
            # distances = []
            # for tf in transforms:
            #     t_translation = np.array([
            #         tf.transform.translation.x,
            #         tf.transform.translation.y,
            #         tf.transform.translation.z
            #     ], dtype=np.float64)
            #     distance = np.linalg.norm(centroid - t_translation)
            #     distances.append(distance)
            # # Select the transform with the smallest distance

            #nearest_tf = transforms[np.argmin(distances)]
            nearest_tf = transforms[0]
            trans = nearest_tf.transform.translation
            rot = nearest_tf.transform.rotation
            gt = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w]).astype(np.float64)
            
            gt[0, 3] = trans.x
            gt[1, 3] = trans.y
            gt[2, 3] = trans.z
            print(trans.x, trans.y, trans.z)

            node.get_logger().info(f"Saving sample for key {key}: {pts.shape[0]}")
            data_file = "data_test"
            os.makedirs(f'/home/ali/Documents/robotic_picker/{data_file}', exist_ok=True)
            np.savez_compressed(f'/home/ali/Documents/robotic_picker/{data_file}/dataset{node.id}.npz',
                                X=pts.astype(np.float64),
                                y=gt.astype(np.float64))
            node.id += 1

def segment_points(masks_list, points_reshaped):
    filtered_points = {}
    for i, mask in enumerate(masks_list):
        mask_bool = mask > 0
        filtered = points_reshaped[mask_bool]
        if len(filtered) > 0:
            # Compute centroid and distances to it
            print(f"number of segmented points {len(filtered)}")
            centroid = np.mean(filtered, axis=0)
            distances = np.linalg.norm(filtered - centroid, axis=1)
            # Use median absolute deviation to set a robust threshold
            median_dist = np.median(distances)
            mad = np.median(np.abs(distances - median_dist))
            # Set threshold as median + 3 * MAD (avoid zero MAD)
            threshold = median_dist + 3 * (mad if mad > 0 else 1e-6)
            # Keep only inliers that are not too far from the centroid
            inliers = filtered[distances <= threshold]
            print(f"number of inliers {len(inliers)}")
            if len(inliers) > 0:
                filtered_points[i] = inliers

    return filtered_points


def transform_points(points, T):
        if points.shape[0] == 0:
            return points
        homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed = (T @ homogeneous.T).T
        return transformed[:, :3]


def process_pointcloud():
    # Convert PointCloud2 to numpy array
    # maybe segmentation should happen here instead of in the model
    
    

    # In process_pointcloud(), insert the following code at the $SELECTION_PLACEHOLDER$ location:
    if node.rgb_image is None or node.pointcloud is None:
        node.get_logger().warn("RGB image or pointcloud not received")
        return
    
    masks_list = get_segmentation_masks()
    if not masks_list:
        return
   
    header_ = node.pointcloud.header
    points_list = [[p['x'], p['y'], p['z']] for p in read_points(node.pointcloud, field_names=("x", "y", "z"), skip_nans=True)]
    points = np.nan_to_num(np.array(points_list), nan=0.0)
    height, width = masks_list[0].shape  # Assuming all masks have same dimensions
    points_reshaped = points.reshape(height, width, -1)  # Preserve xyz channels
    


    filtered_points = segment_points(masks_list, points_reshaped)
 
    # Attempt to look up three transforms
    try:
        tf = node.tf_buffer.lookup_transform("world", "Camera_OmniVision_OV9782_Color", header_.stamp, rclpy.duration.Duration(seconds=1.0))
        # tf2 = node.tf_buffer.lookup_transform("Camera_OmniVision_OV9782_Color", "redStrawberry_01", msg.header.stamp, rclpy.duration.Duration(seconds=1.0))
        # tf3 = node.tf_buffer.lookup_transform("Camera_OmniVision_OV9782_Color", "redStrawberry_04", msg.header.stamp, rclpy.duration.Duration(seconds=1.0))
    except Exception as e:
        node.get_logger().warn("TF lookup failed: %s" % str(e))
        return

    # # Collect transforms in a list
    # #transforms = [tf1, tf2, tf3]
    # transforms = [tf1]
    # save_data(filtered_points, transforms)      

    

    

   
        
    all_points, _ = process_filtered_points(filtered_points)

    if not all_points:
        return

    predictions = []
    net.eval()
    with torch.no_grad():
        for pts in all_points:
            input_cpu = torch.from_numpy(np.expand_dims(pts, axis=0)).float()
            input_tensor = input_cpu.cuda().transpose(2, 1).contiguous()
            result_dict = net(input_tensor, None, is_training=False)
            pred = result_dict['out2'].cpu().numpy()
            predictions.append(pred[0])

    # Compute centroids from the model predictions and decide target vs. obstacles

    # Compute the transformation matrix from the tf transform
    trans = tf.transform.translation
    rot = tf.transform.rotation
    world_T = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    world_T[0, 3] = trans.x
    world_T[1, 3] = trans.y
    world_T[2, 3] = trans.z

    
    # Apply the transformation to all predicted pointclouds
    pointclouds = [transform_points(pred, world_T) for pred in predictions]

    if len(pointclouds) == 0:
        target_points = np.empty((0, 3))
        obstacle_points = np.empty((0, 3))
    else:
        centroids = [np.mean(pc, axis=0) for pc in pointclouds if pc.size > 0]
        if len(centroids) == 0:
            target_points = np.empty((0, 3))
            obstacle_points = np.empty((0, 3))
        else:
            centroids = np.array(centroids)
            distances = np.linalg.norm(centroids, axis=1)
            key_idx = np.argmin(distances)
            key_idx = np.random.choice(len(distances))

            target_points = pointclouds[key_idx]
            other_pcs = [pc for i, pc in enumerate(pointclouds) if i != key_idx]
            obstacle_points = np.concatenate(other_pcs, axis=0) if other_pcs else np.empty((0, 3))

## Convert each point to a geometry_msgs/Point message
        
        if target_points.size > 0:
            # Calculate the center and bounding box dimensions from the target point cloud
            centroid = np.mean(target_points, axis=0)
            min_vals = np.min(target_points, axis=0)
            max_vals = np.max(target_points, axis=0)
            sizes = max_vals - min_vals
     # Prepare a Detection3D message for the target object
            detection = Detection3D()
            detection.header = header_
            detection.bbox.center.position.x = centroid[0]
            detection.bbox.center.position.y = centroid[1]
            detection.bbox.center.position.z = centroid[2]
            detection.bbox.size.x = sizes[0]
            detection.bbox.size.y = sizes[1]
            detection.bbox.size.z = sizes[2]
            # Fixed orientation (example: identity quaternion)
            detection.bbox.center.orientation.w = 0.7071068
            detection.bbox.center.orientation.x = 0.7071068
            detection.bbox.center.orientation.y = 0.0
            detection.bbox.center.orientation.z = 0.0
     # Create a Detection3DArray message and add the detection to it
            detection_array = Detection3DArray()
            detection_array.header = header_
            detection_array.detections.append(detection)
            

     # Initialize the publisher if not already created
            # if not hasattr(node, "target_detections_pub"):
            #     node.target_detections_pub = node.create_publisher(Detection3DArray, "target_detections", 10)
            # node.target_detections_pub.publish(detection_array)

    header = header_
    header.frame_id = "world"
    node.target_msg = pc2.create_cloud_xyz32(header, target_points.tolist())
    node.obstacle_msg = pc2.create_cloud_xyz32(header, obstacle_points.tolist())
    if not hasattr(node, "target_pub"):
        node.target_pub = node.create_publisher(PointCloud2, "target_points", 10)
    if not hasattr(node, "obstacle_pub"):
        node.obstacle_pub = node.create_publisher(PointCloud2, "filtered_pointcloud", 10)
    node.target_pub.publish(node.target_msg)
    node.obstacle_pub.publish(node.obstacle_msg)

    node._publisher_detection.publish(detection_array)
    print(f"published detection array of size {len(detection_array.detections)}")




   
    

def trigger_callback(request, response):
    node.target_msg = None 
    node.obstacle_msg = None
    if not clear_octomap_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().warn("Service /clear_octomap not available!")
    else:
        req = Empty_srv.Request()
        future = clear_octomap_client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
        if future.done() and future.exception() is None:
            node.get_logger().info("Octomap cleared successfully.")
        else:
            node.get_logger().error("Failed to clear octomap or timed out.")
    process_pointcloud()
    response.success = True
    response.message = "Processed pointcloud after clearing octomap."
    return response
 
def timer_callback():
    if node.target_msg is not None and node.obstacle_msg is not None:
        node.target_pub.publish(node.target_msg)
        node.obstacle_pub.publish(node.obstacle_msg)


   
def predict_poses():
    predictions = []
    net.eval()
    with torch.no_grad():
      
        # Directory containing segmented *.ply files
        directory = "segmented_real_pointcloud"
        ply_files = sorted(glob.glob(f"{directory}/*.ply"))
        print(len(ply_files))
        input_pointclouds = []
        predicted_pointclouds = []
        net.eval()
        with torch.no_grad():
            for file in ply_files:
                # Load the point cloud
                pcd = o3d.io.read_point_cloud(file)
                pts = np.asarray(pcd.points)
                
                # Downsample using farthest point sampling if enough points, else repeat points until 512
                
                if pts.shape[0] >= 512:
                    sampled_pcd = pcd.farthest_point_down_sample(512)
                    pts_sampled = np.asarray(sampled_pcd.points)
                else:
                    repeats = int(np.ceil(512 / pts.shape[0]))
                    pts_sampled = np.tile(pts, (repeats, 1))[:512]
                
                if pts_sampled.shape[0] < 512:
                    continue
                print(f"Processing {file} with {pts_sampled.shape[0]} points")
                input_pointclouds.append(pts_sampled)
                
                # Run inference using the loaded network
                input_tensor = torch.from_numpy(np.expand_dims(pts_sampled, axis=0)).float().cuda()
                input_tensor = input_tensor.transpose(2, 1).contiguous()
                result = net(input_tensor, None, is_training=False)
                pred = result['out2'].cpu().numpy()[0]
                predicted_pointclouds.append(pred)

            # Prepare and visualize each pair of point clouds independently
            for inp, pred in zip(input_pointclouds, predicted_pointclouds):
                # Create input point cloud (red)
                pcd_input = o3d.geometry.PointCloud()
                pcd_input.points = o3d.utility.Vector3dVector(inp)
                pcd_input.paint_uniform_color([1, 0, 0])
                
                # Create output point cloud (blue)
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(pred)
                pcd_pred.paint_uniform_color([0, 0, 1])
                
                # Offset the prediction cloud along X for side-by-side visualization
                bounds = pcd_input.get_axis_aligned_bounding_box()
                offset = np.array([bounds.get_extent()[0] + 0.1, 0, 0])
                pcd_pred.translate(offset)
                
                # Visualize the input and prediction clouds together
                o3d.visualization.draw_geometries([pcd_input, pcd_pred])
if __name__ == "__main__":
    

    # PointAttn setup and loading
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs',arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')
    
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
   
    #predict_poses()


    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])


    # Initialize ROS2
    rclpy.init()
    node = Node('pointcloud_processor')
    model_path = 'yolo_model.pt'
    node.yolo_model = YOLO(model_path)
     # sam2 setup        
    checkpoint = "sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    node.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    node.tf_buffer = tf2_ros.Buffer()
    node.tf_listener = tf2_ros.TransformListener(node.tf_buffer, node)
    node.bridge = CvBridge()
    node.id = 0 
    node.rgb_sub = node.create_subscription(Image, "/shoulder_eye/rgb", rgb_callback, 10)
    node.pcd_sub = node.create_subscription(
        PointCloud2,
        '/shoulder_eye/pointcloud',  # Change this to your actual topic name
        pointcloud_callback,
        10
    )
    clear_octomap_client = node.create_client(Empty_srv, '/clear_octomap')
    node.target_msg = None
    node.obstacle_msg = None
    #node.depth_sub = node.create_subscription(Image, "/shoulder_eye/depth", depth_callback, 10)
    node.reset_pub = node.create_publisher(Empty, '/reset', 10)
    node.rgb_image = None
    node.detections = None
    node.pointcloud = None
    
    node.create_service(Trigger, 'process_pointcloud', trigger_callback)
    node._publisher_detection = node.create_publisher(Detection3DArray, 'grasp_poses', 10)
    
    

    node.create_timer(1.0, timer_callback)

    
    print('Waiting for PointCloud2 messages...')
    
    # Spin to receive messages
    rclpy.spin(node)
    
    #test()
    

# def throttled_process_pointcloud(msg):
#     now = node.get_clock().now()
#     if not hasattr(node, 'last_pc_time'):
#         node.last_pc_time = now
#         process_pointcloud(msg)
#     elif (now - node.last_pc_time).nanoseconds / 1e9 >= 4.0:
#         node.last_pc_time = now
#         process_pointcloud(msg)
   

# def depth_callback(data):
#     print("depth callback")
#     try:
#         node.depth_image = node.bridge.imgmsg_to_cv2(data, "32FC1")
#         # Normalize depth image for visualization
#         depth_normalized = cv2.normalize(node.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#         print(depth_normalized.shape)
#         # Display depth image
        
#     except Exception as e:
#         node.get_logger().error("Error converting depth image: %s" % str(e))
