#ARG IMAGE_NAME
#FROM ${IMAGE_NAME}:8.0-devel-ubuntu16.04
FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends libcudnn6=$CUDNN_VERSION-1+cuda8.0 libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 software-properties-common && rm -rf /var/lib/apt/lists/*

#RUN apt-get update
#RUN apt-get install -y --no-install-recommends \
#    apt-transport-https \
#    ca-certificates \
#    curl \
#    software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get update

RUN add-apt-repository -r ppa:mc3man/trusty-media
RUN apt-get update

RUN apt-get install -y python3.6 libpython3.6 python3-pip libcap-dev libsm6 libxrender1 ffmpeg
COPY Codes/requirements.txt /root/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /root/requirements.txt
#RUN pip3 install --index-url=http://mirrors.aliyun.com/pypi/simple/ -r /root/requirements.txt --trusted-host mirrors.aliyun.com

COPY . /root/code