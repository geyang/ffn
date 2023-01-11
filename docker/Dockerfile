FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04
ENV LANG=C.UTF-8

RUN apt-get -y update && apt-get install -y git vim tree curl unzip xvfb patchelf ffmpeg cmake swig
RUN apt-get install -y libssl-dev libcurl4-openssl-dev  # Needed for pyCurl
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev
RUN apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
RUN apt-get install -y qtbase5-dev
RUN apt-get install -y qtdeclarative5-dev
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y wget

# Download mujoco
RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xvf mujoco210-linux-x86_64.tar.gz


ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_PY_MJPRO_PATH /root/.mujoco/mujoco210
ENV MJLIB_PATH /root/.mujoco/mujoco210/bin/libmujoco210.so
# ENV MJKEY_PATH /root/.mujoco/mjkey.txt

RUN apt-get install -y libssl-dev libcurl4-openssl-dev
RUN apt-get update && apt-get install -y python3.8-dev python3-pip
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN python -m pip install --upgrade pip

ENV PIP_NO_CACHE_DIR=off

RUN pip install pytest pytest-forked lz4 pyyaml qt5-py
RUN pip install gym3
RUN pip install gym==0.21.0
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
RUN pip install kornia opencv-python
RUN pip install tensorflow tensorboard
RUN git clone https://github.com/openai/baselines.git
RUN cd baselines && pip install -e .
RUN pip install filelock

#RUN apt-get install software-properties-common
RUN apt-get install -y libgl1-mesa-glx libosmesa6
RUN apt-get install -y libglew-dev
RUN pip install dm-control==0.0.403778684 
RUN pip install gym-dmc==0.2.5
# used for egl rendering
ENV MUJOCO_GL egl
#ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
#:/usr/lib/x86_64-linux-gnu/libGL.so.1

RUN pip install jaynes==0.8.11 ml_logger==v0.8.69 waterbear params-proto==2.9.6 functional-notations --no-cache-dir
# ADD mjkey.txt /root/.mujoco/mjkey.txt
RUN pip install mujoco_py
RUN pip install google-cloud-storage
RUN pip install networkx
RUN pip install --upgrade gym==0.21.0
