FROM ubuntu:18.04
# skip interactive setup when installing packages
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update; \
    apt-get install -y \
    build-essential \
    git \
    git-lfs \
    cmake \
    libopencv-dev \
    libceres-dev; \
    apt-get clean
