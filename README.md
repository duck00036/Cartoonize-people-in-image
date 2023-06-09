# Cartoonize people in image
This repository contains code for converting images of people into cartoon-like illustrations using an instance segmentation model and a cartoonization model. The instance segmentation model is responsible for identifying and separating the people in the input image from the background. This project offers two options for the instance segmentation model: a Mask R-CNN model trained in [my previous work](https://github.com/duck00036/Training-Mask-RCNN-on-custom-dataset-using-pytorch), and a Deeplabv3 model with an ImageNet backbone provided by PyTorch.

The cartoonization model applies a White-box-Cartoonization technique, which stylizes the image in a way that resembles a cartoon drawing,  the implementation origins from [the paper by Xinrui Wang](https://github.com/SystemErrorWang/White-box-Cartoonization). To ensure efficient deployment, this project exports all models to the ONNX format and runs the application on the CPU by default.

Using this repository, you can easily get cartoon versions of the people in your pictures like this:

![a1](https://user-images.githubusercontent.com/48171500/231552353-36036bab-d0ac-4ba2-8536-da01c180a8bd.jpg)

# Installation
To run this project, you will need to install the following packages:
* onnxruntime or (onnxruntime-gpu)
* OpenCV
* NumPy
* tqdm
* matplotlib (Option)
* moviepy (Option)

You can install these packages using pip:
```
pip install onnxruntime numpy matplotlib opencv-python tqdm
```
if you wanna use GPU:
```
pip install onnxruntime-gpu numpy matplotlib opencv-python tqdm moviepy
```
## Docker

Or you can choose to pull the docker I built
```
docker pull public.ecr.aws/v0s3r6q0/cartoonize_people:cpu
```
and tag a new name for further use
```
docker tag public.ecr.aws/v0s3r6q0/cartoonize_people:cpu cartoon:cpu
```

# Usage
To use this repository, you should first clone this repository.
```
git clone https://github.com/duck00036/Cartoonize-people-in-image.git
```
Then download the models:
| mask-rcnn | deeplabv3 | cartoonize model |
|:--------- |:--------- |:---------------- |
| [person_classifier.onnx](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/person_classifier.onnx) | [deeplabv3.onnx](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/deeplabv3.onnx) | [cartoonize.onnx](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/cartoonize.onnx) |

Put the photos you want to cartoonize into the "**input_image**" folder.

If you want to use deeplabv3 as the instance segmentation model, run the script:
```
python trans_deep.py
```
If you want to use mask-rcnn as the instance segmentation model, run the script (this may take longer):
```
python trans_mask.py
```
(Since the model is exported from PyTorch to onnx, there may be some warning messages about unused initializers, but it does not affect the operation)

The output photos will be saved in the "**output_image**" folder.
## with Docker
Create two folders as an input folder and an output folder, and put the photos to be cartoonized into the input folder.

If you want to use deeplabv3 as the instance segmentation model, run the script:
```
docker run --rm -v <input path>:/trans/input_image -v <output path>:/trans/output_image cartoon:cpu python3 trans_deep.py
```
If you want to use mask-rcnn as the instance segmentation model, run the script (this may take longer):
```
docker run --rm -v <input path>:/trans/input_image -v <output path>:/trans/output_image cartoon:cpu python3 trans_mask.py
```
(Since the model is exported from PyTorch to onnx, there may be some warning messages about unused initializers, but it does not affect the operation)

* **input path** is your own input folder's path
* **output path** is your own output folder's path
    
The output photos will be saved in your output folder.

Note: The functions in docker can only cartoonize photos, not videos

## with jupyter notebook
If you want to understand the implementation process or want to make adjustments to the code, [this](trans_onnx.ipynb) notebook may help you.

## with GPU
### cartoonize videos
If you have an onnxruntime-gpu environment, with matched CUDA and cudaa, you can also cartoonize the videos !

(CPU is also available, but it will be extremely slow)

Put the videos you want to cartoonize into the "input_video" folder and make sure you have the moviepy library installed.

Run the script (Considering efficiency, we use deeplabv3 model for instance segmentation):
```
python trans_video.py
```
### cartoonize yourself
And if you wanna try a self cartoonize cam, you can run this script:
```
python selfcam.py
```
If you want to quit, you can press the "q" key to quit.

# Demo
## Photo
![p1](https://user-images.githubusercontent.com/48171500/231575267-77ab314e-58a2-47a7-aac3-1ebc0c9e3245.jpg)
![p2](https://user-images.githubusercontent.com/48171500/231575289-d0ee4ead-8fdc-40ed-9caa-8830a11837ac.jpg)

## Video
<img src="https://user-images.githubusercontent.com/48171500/231588071-181ff972-461a-44b7-ac8c-5016bd58b31c.gif" width="400"/><img src="https://user-images.githubusercontent.com/48171500/231588153-e857b3e0-b247-4d6e-bc0e-c1cfe1a83d7b.gif" width="400"/>

<img src="https://user-images.githubusercontent.com/48171500/231588176-42052910-394f-4997-a957-6b26fdc87a35.gif" width="400"/><img src="https://user-images.githubusercontent.com/48171500/231588191-515171ff-a087-4cde-854c-c513cf2fbcee.gif" width="400"/>




# reference

[PyTorch deeplabv3 documentation](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)

[Tensorflow implementation for CVPR2020 paper “Learning to Cartoonize Using White-box Cartoon Representations”](https://github.com/SystemErrorWang/White-box-Cartoonization)
