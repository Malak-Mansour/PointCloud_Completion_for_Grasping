import glob
import os
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
 
# Paths
base_path = "log/PointAttN_cd_debug_pcn/all"
import glob, os
import open3d as o3d
import numpy as np


def cal_normal(pcd, radius=0.03, max_nn=30):
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pcd)
    
    _pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # o3d.geometry.estimate_normals(_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(_pcd.normals)
    return normals
 
 
def visualize_comparison(src_pcd, src_pcd_inter, batch_idx, sample_idx):
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Batch {batch_idx} Sample {sample_idx}")
    
    # Add source point cloud (red)
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_pcd)
    pcd_src.paint_uniform_color([1, 0, 0])  # Red
    vis.add_geometry(pcd_src)
    
    # Add interpolated point cloud (blue)
    pcd_inter = o3d.geometry.PointCloud()
    pcd_inter.points = o3d.utility.Vector3dVector(src_pcd_inter)
    pcd_inter.paint_uniform_color([0, 0, 1])  # Blue
    vis.add_geometry(pcd_inter)
    
    # Run visualization
    vis.run()
    vis.destroy_window()
 
 
# Process each file in the directory
 
files = glob.glob(os.path.join(base_path, "batch*_sample*_data.npz"))
 
for file_path in files:
    # Extract batch and sample numbers from filename
    filename = os.path.basename(file_path)
    batch_idx = int(filename.split('_')[0].replace('batch', ''))
    
    sample_idx = int(filename.split('_')[1].replace('sample', ''))
    
    # Load the data
    data = np.load(file_path)
    src_pcd = data['src_pcd']
    src_pcd_pred = data['xyz']
    unique_points = np.unique(src_pcd_pred, axis=0)
    print(f"Number of unique points: {len(unique_points)}, Total points: {len(src_pcd_pred)}")
    #FPS on src_pcd_inter to get 512 points
    pcd_inter = o3d.geometry.PointCloud()
    pcd_inter.points = o3d.utility.Vector3dVector(src_pcd_pred)
    if len(src_pcd_pred) > 512:
        downsampled_pcd_inter = pcd_inter.farthest_point_down_sample(512)
        src_pcd_pred = np.asarray(downsampled_pcd_inter.points)
    else:
        src_pcd_pred = np.asarray(pcd_inter.points)
        repeats_needed = int(np.ceil(512 / len(src_pcd_pred)))
        src_pcd_pred = np.tile(src_pcd_pred, (repeats_needed, 1))[:512]
    
    print(f"Visualizing batch {batch_idx}, sample {sample_idx}")
    #src_pcd_normal = cal_normal(src_pcd_pred)
    visualize_comparison(src_pcd, src_pcd_pred, batch_idx, sample_idx)
    
    # Optional: wait for user input before showing next visualization
    #input("Press Enter to continue to next sample...")
    
    # # Prepare data for saving in the desired format
    # output_data = {
    #     "src_pcd": src_pcd_pred,  # Use processed 'src_pcd_inter' as 'src_pcd'
    #     "src_pcd_normal": src_pcd_normal,
    #     "model_pcd": data['model_pcd'],
    #     "model_pcd_normal": data['model_pcd_normal'],
    #     "model_pcd_transformed": data['model_pcd_transformed'],
    #     "transform_gt": data['transform_gt']
    # }
    
    # Construct the output file path in data_dir
    
    
    
    # Save the modified data to the new file
    # print(f"Saving modified data to {output_file_path}")
    # np.savez(output_file_path, **output_data)
 
 