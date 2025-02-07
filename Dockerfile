FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

RUN apt update && apt upgrade -y && \
    apt install -y \
        build-essential \
        g++ \
        wget \
        python3-pip \
        python3-dev \
        python3-setuptools \
        libboost-python-dev \
        libboost-thread-dev \
        cuda-toolkit-12-3 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

RUN wget https://files.pythonhosted.org/packages/61/69/f53a6624def08348778a7407683f44c2a9adfdb0b68b9a45f8213ff66c9d/pycuda-2024.1.2.tar.gz && \
    tar -xvzf pycuda-2024.1.2.tar.gz && \
    cd pycuda-2024.1.2 && \
    ./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-py27 --boost-thread-libname=boost_thread && \
    make -j 4 && \
    python setup.py install && \
    pip install . && \
    cd .. && \
    rm -rf pycuda-2024.1.2 pycuda-2024.1.2.tar.gz
