# [CoRL 2025] BEVCalib: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representation

[![arXiv](https://img.shields.io/badge/arXiv-2506.02587-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2506.02587) [![Website](https://img.shields.io/badge/Website-BEVCalib-blue?style=for-the-badge)](https://cisl.ucr.edu/BEVCalib) [![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/cisl-hf/BEVCalib) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) [![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

<hr style="border: 2px solid gray;"></hr>

## Getting Started

### Prerequistes
First create a conda environment:
```bash
conda env create -n bevcalib python=3.11
conda activate bevcalib
pip3 install -r requirements.txt
```

The code is built with following libraries:

- Python = 3.11
- Pytorch = 2.6.0
- CUDA = 11.8
- cuda-toolkit = 11.8
- [spconv-cu118](https://github.com/traveller59/spconv)
- OpenCV
- pandas
- open3d
- transformers
- [deformable_attention](https://github.com/lucidrains/deformable-attention)
- tensorboard
- wandb
- pykitti

We recommend using the following command to install cuda-toolkit=11.8:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

After installing the above dependencies, please run the following command to install [bev_pool](https://github.com/mit-han-lab/bevfusion) operation
```bash
cd ./core/img_branch/bev_pool && python setup.py build_ext --inplace
```

We also provide a [Dockerfile](Dockerfile/Dockerfile) for easy setup, please execute the following command to build the docker image and install cuda extensions:
```bash
docker build -f Dockerfile/Dockerfile -t bevcalib .
docker run --gpus all -it -v$(pwd):/workspace bevcalib
### In the docker, run the following command to install cuda extensions
cd ./core/img_branch/bev_pool && python setup.py build_ext --inplace
```

## Dataset Preparation
### KITTI-Odometry
We release the code to reproduce our results on the KITTI-Odometry dataset. Please download the KITTI-Odometry dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). After downloading the dataset, the directory structure should look like
```tree
kitti-odometry/
├── sequences/         
│   ├── 00/            
│   │   ├── image_2/  
│   │   ├── image_3/   
│   │   ├── velodyne/
│   │   └── calib.txt 
│   ├── 01/
│   │   ├── ...
│   └── 21/
│       └── ...
└── poses/            
    ├── 00.txt        
    ├── 01.txt
    └── ...
```

### CalibDB
Coming soon!

## Pretrained Model
We release our pretrained model on the KITTI-Odometry dataset. We provide two ways to download our models.
### Google cloud
Please find the pretrained model from [Google Drive](https://drive.google.com/drive/folders/157yrmTHntjqF5nP_cVo22BfeVPhiz2ni?usp=share_link) and place it in the `./ckpt` directory. For your convenience, you can also run `pip3 install gdown` and run the following command to download the KITTI checkpoint in the command line.

```bash
gdown https://drive.google.com/uc\?id\=1gWO-Z4NXG2uWwsZPecjWByaZVtgJ0XNb
```
### Hugging face
We also release our pretrained model on [Hugging Face page](https://huggingface.co/cisl-hf/BEVCalib). You should download huggingface-cli by `pip install -U "huggingface_hub[cli]"` and then download the pretrained model by running the following command:
```bash
huggingface-cli download cisl-hf/BEVCalib --revision kitti-bev-calib --local-dir YOUR_LOCAL_PATH
```

### Environment Setup
Before running any scripts, set the `PYTHONPATH` to the repository root so that shared modules in `core/` can be found:
```bash
export PYTHONPATH=$(pwd)
```

## Evaluation
Please run the following command to evaluate the model:
```bash
python kitti-bev-calib/inference_kitti.py \
         --log_dir ./logs/kitti \
         --dataset_root YOUR_PATH_TO_KITTI/kitti-odemetry \
         --ckpt_path YOUR_PATH_TO_KITTI_CHECKPOINT/ckpt/ckpt.pth \
         --angle_range_deg 20.0 \
         --trans_range 1.5
```

## Training
We provide instructions to reproduce our results on the KITTI-Ododemetry dataset. Please run: 
```bash
python kitti-bev-calib/train_kitti.py --log_dir ./logs/kitti \
        --dataset_root YOUR_PATH_TO_KITTI/kitti-odemetry \
        --save_ckpt_per_epoches 40 --num_epochs 500 --label 20_1.5 --angle_range_deg 20 --trans_range 1.5 \
        --deformable 0 --bev_encoder 1 --batch_size 16 --xyz_only 1 --scheduler 1 --lr 1e-4 --step_size 80
```
You can change `--angle_range_deg` and `--trans_range` to train under different noise settings. You can also try to use `--pretrain_ckpt` to load a pretrained model for fine-tuning on your own dataset.

## Acknowledgement
BEVCalib appreciates the following great open-source projects: [BEVFusion](https://github.com/mit-han-lab/bevfusion?tab=readme-ov-file), [LCCNet](https://github.com/IIPCVLAB/LCCNet), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [spconv](https://github.com/traveller59/spconv), and [Deformable Attention](https://github.com/lucidrains/deformable-attention).

## Citation
```
@inproceedings{bevcalib,
      title={BEVCALIB: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representations}, 
      author={Weiduo Yuan and Jerry Li and Justin Yue and Divyank Shah and Konstantinos Karydis and Hang Qiu},
      booktitle={9th Annual Conference on Robot Learning},
      year={2025},
}
```
