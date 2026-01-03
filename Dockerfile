FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin

RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get upgrade -y && \
    pip install wandb