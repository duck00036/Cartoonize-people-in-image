FROM ubuntu:20.04

RUN apt update
RUN apt install -y tzdata
RUN apt install -y python3-pip git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

RUN mkdir trans
WORKDIR /trans
RUN python3 -m pip install --upgrade pip wheel
RUN pip install opencv-python onnxruntime numpy tqdm --target=/trans

COPY trans_deep.py trans_mask.py inference.py cartoonize.onnx deeplabv3.onnx person_classifier.onnx /trans/






