FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y python3-pip

RUN mkdir trans
WORKDIR /trans
RUN pip install opencv-python-headless onnxruntime numpy tqdm --target=/trans

COPY trans_deep.py trans_mask.py inference.py cartoonize.onnx deeplabv3.onnx person_classifier.onnx /trans/