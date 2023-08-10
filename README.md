# Reflash-Dropout-in-Image-Super-Resolution
(CVPR2022) Reflash Dropout in Image Super-Resolution

[Paper link](https://arxiv.org/pdf/2112.12089.pdf)


[Talk](https://www.bilibili.com/video/BV1gY4y1k7fU?spm_id_from=333.337.search-card.all.click) (Chinese, start from 26:35)

One line of dropout brings more improvement than ten times of model parameters (SRResNet && RRDB).


<img src="https://raw.githubusercontent.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution/main/result.png" width="550px">


## Abstract

Dropout is designed to relieve the overfitting problem in high-level vision tasks but is rarely applied in low-level vision tasks, like image super-resolution (SR). As a classic regression problem, SR exhibits a different behaviour as high-level tasks and is sensitive to the dropout operation. However, in this paper, we show that appropriate usage of dropout benefits SR networks and improves the generalization ability. Specifically, dropout is better embedded at the end of the network and is significantly helpful for the multi-degradation settings. This discovery breaks our common sense and inspires us to explore its working mechanism. We further use two analysis tools -- one is from recent network interpretation works, and the other is specially designed for this task. The analysis results provide side proofs to our experimental findings and show us a new perspective to understand SR networks.

The core code is adding nn.functional.dropout(or dropout2d) into RealESRNet (https://github.com/xinntao/Real-ESRGAN)

## Installation

1. Install dependent packages 
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install basicsr`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

2. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution.git
cd Reflash-Dropout-in-Image-Super-Resolution
```

## How to test Real-SRResNet or Real-RRDB (w/ or w/o) dropout

**Some steps require replacing your local paths.**

1. Move to experiment dir.
```
cd Real-train
```

2. Download the testing datasets (Set5, Set14, B100, Manga109, Urban100) and move them to `./dataset/benchmark`.
[Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) or [Baidu Drive](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (Password: basr) .

3. Add degradations to testing datasets.
```
cd ./dataset
python add_degradations.py
```

4. Download [pretrained models](https://drive.google.com/drive/folders/1NcNHbsGtD0OHuAf_ATACmZ_cTikL7bB3?usp=sharing) and move them to  `./pretrained_models/` folder. 

   To remain the setting of Real-ESRGAN, we use the GT USM (sharpness) in the paper. But we also provide the models without USM, the improvement is basically same.

5. Run the testing commands.
```
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realrrdbnet.yml
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realrrdbnet_withdropout.yml
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realsrresnet.yml
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realsrresnet_withdropout.yml
```
6. The output results will be sorted in `./results`. 

## How to train Real-SRResNet or Real-RRDB (w/ or w/o) dropout

**Some steps require replacing your local paths.**

1. Move to experiment dir.
```
cd Real-train
```

2. Download the training datasets([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)), move it to `./dataset` and validation dataset(Set5), move it to `./dataset/benchmark`.

3. Generate sub-images and meta-info for training. 
```
cd ./dataset
python extract_subimages_train.py
python generate_meta_info.py
```

4. Run the training commands.
```
cd codes
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realrrdbnet.yml --launcher pytorch --auto_resume
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realrrdbnet_withdropout.yml --launcher pytorch --auto_resume
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realsrresnet.yml --launcher pytorch --auto_resume
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realsrresnet_withdropout.yml --launcher pytorch --auto_resume
```
5. The experiments will be sorted in `./experiments`. 

## How to generate channel saliency map (CSM)

**Some steps require replacing your local paths.**

1. Move to CSM dir.
```
cd CSM
```
2. Download your testing datasets (Here we take Set5 as an example). 

3. Generate attribution map, feature map and ablation results. 
```
python generate_map.py
```
4. The output results will be sorted in `./results`. 


### Code function
1. ModelZoo. 
Load the pretrained models (To generate CSM).

2. SaliencyModel.
Attribution method of attributing the prediction of a deep network to its input features. A process of calculating gradient and backpropagation.

## How to generate deep degradation representation (DDR)
The code of DDR (https://arxiv.org/pdf/2108.00406.pdf) will be released these days by https://github.com/lyh-18 in his projects.
We will update the code after releasing.

## Citation
```
@inproceedings{kong2022reflash,
  title={Reflash dropout in image super-resolution},
  author={Kong, Xiangtao and Liu, Xina and Gu, Jinjin and Qiao, Yu and Dong, Chao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6002--6012},
  year={2022}
}
```
