FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

RUN apt update && apt upgrade -y
RUN apt install build-essential
RUN apt install -y g++
RUN apt install -y wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-ubuntu2204-12-5-local_12.5.0-555.42.02-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-5-local_12.5.0-555.42.02-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-5

RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2404-9.3.0_1.0-1_amd64.deb
RUN dpkg -i cudnn-local-repo-ubuntu2404-9.3.0_1.0-1_amd64.deb
RUN cp /var/cudnn-local-repo-ubuntu2404-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cudnn

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt 

RUN apt-get update && apt-get install -y build-essential python3-dev python3-setuptools libboost-python-dev libboost-thread-dev

RUN wget https://files.pythonhosted.org/packages/61/69/f53a6624def08348778a7407683f44c2a9adfdb0b68b9a45f8213ff66c9d/pycuda-2024.1.2.tar.gz && \
    tar -xvzf pycuda-2024.1.2.tar.gz && \
    cd pycuda-2024.1.2 && \
    ./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-py27 --boost-thread-libname=boost_thread && \
    make -j 4 && \
    python setup.py install && \
    pip install .