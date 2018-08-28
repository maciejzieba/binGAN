FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y vim curl


## prepare conda
ENV CONDA_DIR /opt/conda

ENV PATH $CONDA_DIR/bin:$PATH


RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp $CONDA_DIR \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=2 \
    && conda update conda


ENV NB_USER lasagne
ENV NB_UID 1000


RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src


## install lasagne
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
RUN git clone https://github.com/maciejzieba/binGAN


CMD ["/bin/bash"]