# modified_refinedet

#### submit to acmmm19

#### Abstract
In recent years, the performance of object detection algorithms based on convolutional neural networks (CNNs) has dramatically improved with the introduction of object detectors using deep neural network structures. With the improvements in the object detection networks, several variations of object detection algorithms have been proposed. However, the performance evaluation of most models has focused on detection accuracy, and the performance verification is based on high-performance high-end GPU hardware. In this paper, we propose a real-time object detector that guarantees accuracy in real-time embedded environments to overcome the problem of object detection. The proposed model utilizes the basic head structure of the RefineDet model, which is a variant of the single shot object detector (SSD). In order to ensure real-time performance, CNN models with relatively shallow layers or fewer parameters have been used as the backbone structure. In addition to the basic VGGNet and ResNet structures, various backbone structures such as MobileNet, Xception, ResNeXt, Inception-SENet, and SE-ResNeXt have been used for this purpose. Successful training of object detection networks was achieved through an appropriate combination of intermediate layers. The accuracy of the proposed detector was determined by applying the evaluation method of MS-COCO 2017 object detection dataset and the inference speed on the NVIDIA Drive PX2 and Jetson Xaviers boards were tested to verify real-time performance in the embedded systems. The results of the analyses show that the proposed model ensures balanced performance in terms of accuracy and inference speed in the embedded system environment. In addition, unlike the high-end GPUs, the use of embedded GPUs involves certain additional concerns for efficient inference, which have been identified in this study. The codes and models are publicly available on [the web (link)](https://github.com/mkchoi-0323/modified_refinedet/).

### 1. Installation

#### 1.1 Requirement
- Ubuntu 16.04
- Caffe
- Python2
- CUDA 9.0

#### 1.2 ReifineDet
- Install original RefineDet source from [git (link)](https://github.com/sfzhang15/RefineDet)
  git clone https://github.com/sfzhang15/RefineDet.git
- Fllowing step to train and test with original RefineDet using COCO dataset [git (link)](http://cocodataset.org/#home)

#### 1.3 Modified RefineDet
- Rebuild caffe for original RefineDet with new layers (depthwise convolution and axpy laer)
