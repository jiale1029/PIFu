FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 as runtime

COPY . /app
WORKDIR /app

RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install -y libopenexr-dev openexr \
  && apt-get install -y libglib2.0-0 \
  && apt-get install -y freeglut3-dev \
  && apt-get install -y libgl1-mesa-dri libegl1-mesa libgbm1 \
  && rm -rf /var/lib/apt/lists/*

# install miniconda
SHELL [ "/bin/bash", "--login", "-c"]
ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p $CONDA_DIR && rm ~/miniconda.sh
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# build the conda environment
ENV ENV_PREFIX $PWD/env
RUN conda env create -f /app/environment.yml
RUN /bin/bash -c "source activate PIFu_FYP"

# set the environment for EGL
ENV PYOPENGL_PLATFORM=egl

CMD ["nvidia-smi"]
