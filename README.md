# binGAN

This code implement the binGAN method, that was presented in 
*Zieba M., Semberecki P. El-Gaaly T., Trzcinski T.. "BinGAN: Learning Compact Binary Descriptors with a Regularized GAN."* [arxiv](https://arxiv.org/pdf/1806.06778.pdf) 

Please watch our video [here](https://youtu.be/DpYdhhQF0f8)

The training code requires [Lasagne](http://lasagne.readthedocs.io/en/latest/). Using GPU is highly advised.


## Docker setup


Go into the Github repository directory.

To build Docker image, perform commands:

`docker build  --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t lasagne_img .`

To run docker container instance with GPU support:

`nvidia-docker run  -it --name lasagne_gpu --user lasagne -v $(pwd):/home/lasagne/host/binGAN lasagne_img bash`

The docker container already has the git repository, however you can mount you own host directory by adding `-v ` option, 
for example `/home/user/binGAN:/home/lasagne/host/binGAN` to have results/datasets saved independently to container.

### Running the code

The procedure of training binGan is run with `train_model.py` script. 

There are two datasets for two tasks in which binGAN is evaluated:
- Brown dataset (image matching)
- Cifar10 (image retrieval)    

Selecting the proper dataset is performed in `settings.py`, by setting proper value of `dataset_type` parameter. In this file it is possible also to set various type of training hyperparameters described in details in the [paper](https://arxiv.org/pdf/1806.06778.pdf).

Brown dataset, which is composed of three subsets: `yosemite`, `notredame` and `liberty`. It is crucial to specify, which training and validation set by setting proper values to parameters: `data_name` and `test_data`. 

It is crucial to set proper value of binary features (`num_features`), while switching between datasets.

The quality of image matching on Brown dataset is evaluated by script `test_brown.py`

The quality of image retrieval on Cifar10 dataset is evaluated by script `test_cifar10.py` 




