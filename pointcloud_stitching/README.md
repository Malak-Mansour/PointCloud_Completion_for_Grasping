### Pointcloud stitching with TEASER to extract transformation matrix:

1. From TEASER++: https://github.com/MIT-SPARK/TEASER-plusplus
2. Follow the Minimal Python 3 example instructions
3. In examples/teaser_python_ply/teaser_python_ply.py, replace   dst_cloud = src_cloud.transform(T) with     dst_cloud = o3d.io.read_point_cloud("/home/netbot/Documents/Malak/sam2/sam2/segmented_pointcloud/segmented_0002.ply")
4. Run examples/teaser_python_ply/teaser_python_ply.py to extract the transformation matrix (estimated rotation and translation) between segmented_0001 and segmented_0002
5. Use this in the pointcloud_stitching.ipynb to stitch the image with the transformation matrix

Modification to TEASER-plusplus repo: add stitch_and_visualize.ipynb to \TEASER-plusplus\examples\stitch_and_visualize.ipynb