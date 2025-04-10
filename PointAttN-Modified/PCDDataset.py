import torch, os, numpy as np
# Create train loader from processed dataset files
class PCDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_files = [f for f in os.listdir(data_dir) if f.startswith('processed_dataset_')]
        self.data_dir = data_dir
            
    def __len__(self):
        return len(self.data_files)
            
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)
        # Expected keys in npz files:
        # - src_pcd, src_pcd_normal
        # - model_pcd, model_pcd_normal
        # - transform_gt
        return {
            'src_pcd': torch.from_numpy(data['src_pcd']).float(),
            #'src_pcd_normal': torch.from_numpy(data['src_pcd_normal']).float(),
            #'model_pcd': torch.from_numpy(data['model_pcd']).float(),
            #'model_pcd_normal': torch.from_numpy(data['model_pcd_normal']).float(),
            'model_pcd_transformed': torch.from_numpy(data['model_pcd_transformed']).float(),
            #'transform_gt': torch.from_numpy(data['transform_gt']).float()
        }




    