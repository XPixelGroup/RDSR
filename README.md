# Reflash-Dropout-in-Image-Super-Resolution
(CVPR2022) Reflash Dropout in Image Super-Resolution

Paper link: https://arxiv.org/pdf/2112.12089.pdf


## Dependencies

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.0](https://pytorch.org/)


# Codes 
- The core code is adding nn.functional.dropout(or dropout2d) into RealESRNet (https://github.com/xinntao/Real-ESRGAN).


## How to generate channel saliency map (CSM)
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution.git
cd Reflash-Dropout-in-Image-Super-Resolution
```
2. Download your testing datasets (Here we take Set5 as an example). 

3. Generate attribution map, feature map and ablation results. 
```
python generate_map.py
```
4. The output results will be sorted in `./results`. 


## Code function
1. ModelZoo. 
Load the pretrained models (To generate CSM).

2. SaliencyModel.
Attribution method of attributing the prediction of a deep network to its input features. A process of calculating gradient and backpropagation.