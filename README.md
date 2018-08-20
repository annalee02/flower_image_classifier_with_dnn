# AI Programming with a Python Project
## Flower Image Classifier

This is part of Udacity's Data Scientist Nanodegree program. In this project, I developed code for an image classifier built with **PyTorch**, then converted it into a command line application.

### Structure
* Part 1 - Developing an image classifier <br/>
: I worked through a Jupyter notebook to implement an image classifier with **PyTorch**. 
* Part 2 - Building the command line application <br/>
: After I've built and trained a **Deep Neural Network (DNN)** on the flower data set, I've converted it into an application that others can use. My application is a pair of Python scripts that run from the command line. For testing, you should use the checkpoint I saved in the first part.

### Specifications
* `train.py`: To train a new network on a dataset and save the model as a checkpoint. It prints out training loss, validation loss, and validation accuracy as the network trains. <p> 
* `predict.py`: To use a trained network to predict the class for an input image. It passes in a single image /path/to/image and returns the flower name and class probability.


### Data
The dataset consists of 102 different types of flowers, where there ~20 images per flower to train on. I used my trained classifier to see if it can predict the type for new images of the flowers. <p> 
The data is quite large, so I needed to utilize the GPU enabled workspaces to complete this project. Running on local CPU will likely not work well.
