FROM nvcr.io/nvidia/pytorch:21.12-py3
LABEL maintainer="Brad Larson"

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt-get install -y libsm6 libxext6 ffmpeg # required by opencv-python==4.4.0.42
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y && \
    apt-get install -y libgl1-mesa-glx wget ffmpeg

RUN pip install --upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        numpy \
        opencv-python \
        minio \
        tqdm \
        natsort \
        debugpy \
        path \
        matplotlib \
        torch \
        torchvision==0.7.0 \
        tensorboard \
        tensorboardX \
        torch_tb_profiler \
        scipy \
        scikit-image \
        scikit-learn \
        apex \
        wget \
        configparser \
        pycocotools \
        prettytable \
        onnx \
        onnxruntime-gpu \
        pycuda \
        PyYAML \
        mlflow \
        pymlutil

RUN echo 'alias py=python' >> ~/.bashrc

WORKDIR /app
ENV LANG C.UTF-8
# port 6006 exposes tensorboard
EXPOSE 6006 
# port 3000 exposes debugger
EXPOSE 3000

# Launch training
RUN ["/bin/bash"]
