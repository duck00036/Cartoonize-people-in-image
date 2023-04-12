# Cartoonize people in image
This repository contains code for converting images of people into cartoon-like illustrations using an instance segmentation model and a cartoonization model. The instance segmentation model is responsible for identifying and separating the people in the input image from the background. This project offers two options for the instance segmentation model: a Mask R-CNN model trained in [my previous work](https://github.com/duck00036/Training-Mask-RCNN-on-custom-dataset-using-pytorch), and a Deeplabv3 model with an ImageNet backbone provided by torchvision.

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
If you want to use mask-rcnn as the instance segmentation model, run the script (this might take longer time):
```
python trans_mask.py
```
The output photos will be saved in the "**output_image**" folder.
## with Docker
Create two folders as an input folder and an output folder, and put the photos to be cartoonized into the input folder.

If you want to use deeplabv3 as the instance segmentation model, run the script:
```
docker run --rm -v <input path>:/trans/input_image -v <output path>:/trans/output_image cartoon:cpu python3 trans_deep.py
```
If you want to use mask-rcnn as the instance segmentation model, run the script (this might take longer time):
```
docker run --rm -v <input path>:/trans/input_image -v <output path>:/trans/output_image cartoon:cpu python3 trans_mask.py
```
* **input path** is your own input folder's path
* **output path** is your own output folder's path
    
The output photos will be saved in your ouput folder.
## with jupyter notebook
If you want to understand the implementation process or want to make adjustments to the code, [this](trans_onnx.ipynb) notebook may help you.
## with GPU
