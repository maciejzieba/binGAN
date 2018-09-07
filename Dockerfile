FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y vim curl


## Set Conda Paths
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

## Download MiniConda with Python 2
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp $CONDA_DIR \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=2 \
    && conda update conda


## User name
ENV USER_NAME lasagne

## For generating proper file permission, the UID should be the same as the host user
ARG USER_UID
ARG USER_GID

## Creating user group
RUN groupadd $USER_NAME -g $USER_GID

## Creating non-root user in container
RUN useradd -m -s /bin/bash -N -u $USER_UID -g $USER_GID $USER_NAME && \
    chown $USER_NAME $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $USER_NAME /src



## Install lasagne
RUN conda install -y -c conda-forge matplotlib
RUN conda install -y scikit-learn
RUN conda install theano pygpu
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
RUN conda clean -yt

RUN apt-get install -y git


USER lasagne
WORKDIR /home/lasagne
RUN echo "export MKL_THREADING_LAYER=GNU" >> .bashrc
RUN echo "export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH" >> ~/.bashrc

ADD theanorc .theanorc

RUN mkdir -p /home/lasagne/host/binGAN/

WORKDIR /home/lasagne/host/binGAN/

CMD ["/bin/bash"]