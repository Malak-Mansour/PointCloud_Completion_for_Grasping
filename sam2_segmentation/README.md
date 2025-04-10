### Segmenting pointcloud with SAM2: 

1. From sam2: https://github.com/facebookresearch/sam2
2. conda create --name sam2 python=3.10
3. conda activate sam2
4. pip install torch==2.5.1
5. pip install torchvision==0.20.1
6. git clone https://github.com/facebookresearch/sam2.git
7. cd sam2
8. pip install -e .
9. pip install -e ".[notebooks]"
10. add the 3 files in sam2_additional_files of this repo (segment_pointcloud.ipynb, yolo_model.pt, sam2_hiera_small.pt) into your sam2/sam2 folder
11. run segment_pointcloud.ipynb to segment pointcloud input using the segmented images

Modification to original SAM2 repo: add /sam2/sam2/segment_pointcloud.ipynb, /sam2/sam2/filter_points.py (use yolo.detect instead of yolo.track), sam2/sam2/transform_model_to_real.py

and the weights /sam2/sam2/sam2_hiera_small.pt, /sam2/sam2/yolo_model.pt (https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ali_abouzeid_mbzuai_ac_ae/EsXmWhZEZTxJhKAuM2zVgiABcSbVgcHVQ8_1HYrLKD-juQ?e=Gb4yHe)