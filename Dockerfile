FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

RUN apt update && apt upgrade -y && \
    apt install -y build-essential && \
    apt install -y g++ && \
    apt install -y wget

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt 

FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y build-essential python3-dev python3-setuptools libboost-python-dev libboost-thread-dev

RUN wget https://files.pythonhosted.org/packages/61/69/f53a6624def08348778a7407683f44c2a9adfdb0b68b9a45f8213ff66c9d/pycuda-2024.1.2.tar.gz && \
    tar -xvzf pycuda-2024.1.2.tar.gz && \
    cd pycuda-2024.1.2 && \
    ./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-py27 --boost-thread-libname=boost_thread && \
    make -j 4 && \
    python setup.py install && \
    pip install .