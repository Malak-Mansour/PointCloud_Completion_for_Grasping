We modified PointAttn from https://github.com/ohhhyeahhh/PointAttN


# Environment setup

## Install related libraries

```
conda install -n base conda-libmamba-solver #might be needed if the default solver is too slow
conda config --set solver libmamba


conda create --name pointAttn python=3.10 pip
conda activate pointAttn


cd PointAttN-Modified
pip install -r requirements.txt


conda install nvidia/label/cuda-12.4.0::cuda-nvcc 
conda install cuda-toolkit=12.4.0 -c nvidia 
conda install nvidia/label/cuda-12.4.0::cuda-toolkit


conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```



## Set environment variables
```
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

PyTorch Version: 2.5.1
CUDA Version: 12.4
numpy: 1.24.2
Python: 3.10.16


## Compile Pytorch 3rd-party modules
Compile Pytorch 3rd-party modules [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) and [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion). A simple way is using the following command:

```
cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install
 

cd utils/pointnet2_ops_lib
python setup.py install


cd ../..
```

# Choosing the dataset
Choose either data_sim or data_mix in `train.py` and `test_pcn.py`


# Train, Test, and Visualize
## Train a model

To train the PointAttN model, change the trial mode that you want to run (baseline, downsampled, or with SVD for generation) from the **models** folder by changing the file name to PointAttn only without the mode identifier in the file name, then run:

```
python train.py -c PointAttN.yaml
```


## Test for paper result

To test PointAttN, add log file path of the cd loss best reported at the latest epoch to `cfgs/PointAttN.yaml ` , run:

```
python test_pcn.py -c PointAttN.yaml
```


## Visualizing the 3D pointcloud completion
visualize the sim run using visualize_predictions_malak.py, then delete the `all` folder after ur done