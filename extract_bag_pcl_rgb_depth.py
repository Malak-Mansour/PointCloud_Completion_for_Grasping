import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create folders if they don't exist
os.makedirs("strawberry_extracted_rgb", exist_ok=True)
os.makedirs("strawberry_extracted_depth", exist_ok=True)
os.makedirs("strawberry_extracted_pointcloud", exist_ok=True)

# Initialize the pipeline and configure it to read from a .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("strawberry.bag")
pipeline.start(config)

# Get the depth sensor's depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Frame counter for saving images
frame_counter = 0

try:
    while frame_counter <= 17:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Normalize depth image to 8-bit for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
        
        # Save images using OpenCV
        cv2.imwrite(f"strawberry_extracted_rgb/color_frame_{frame_counter:04d}.png", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"strawberry_extracted_depth/depth_frame_{frame_counter:04d}.png", depth_colormap)
        
        # Extract and save point cloud
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        pc.map_to(color_frame)
        
        np.savetxt(f"strawberry_extracted_pointcloud/pointcloud_{frame_counter:04d}.txt", vtx)
        
        frame_counter += 1

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    pipeline.stop()
    print("Pipeline stopped.")
