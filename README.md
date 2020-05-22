# EE559 - Deep Learning (EPFL), Spring 2020, Project 1: "Classification, weight sharing, auxiliary losses"

The objective of this project is to test different architectures to compare two digits visible in a
two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an
auxiliary loss to help the training of the main objective.

To run the project: *Python test.py*

Training time on the EPFL Deep Learning VM: 10-15 seconds per training with a CNN-based network, 2-3 s with a MLP-based network.
(which gives trial times of about 2 min for CNN networks, and 30 s for MLP networks (for 10 trials)).

In test.py, different modes can be executed: 'train' for a single training, 'trial' for a N-round trial.
Within the 'train' mode, visualizing training loss/accuracy graphs can be done by setting the 'plotting' flag to True.

In this project, four model architectures were implemented in the **models/** folder
* MLP, a baseline multi-layer perceptron
* CNN, a baseline convolutional neural network
* SiameseMLP, the Siamese version of MLP
* SiameseCNN, the Siamese version of CNN

With Siamese networks, the *alpha* coefficient for auxiliary loss can be used. $`alpha=0.5`$ or $`alpha=1`$ generated results approaching 92% of test accuracy.

Description of other files/folders:
* *dlc_practical_prologue.py* contains functions to create pairs from MNIST data
* *utils.py* contains helpers for loading and processing data from *dlc_practical_prologue.py*, as well as initializing weights and creating plots.
* *train.py* contains functions necessary for training, optimizing hyperparameters and performing N-round trials.
* *metrics.py* contains metric functions (only accuracy here)
* **img/** contains images used in the report
* **data/** contains MNIST data from the MNIST handwritten digit database (http://yann.lecun.com/exdb/mnist/)

