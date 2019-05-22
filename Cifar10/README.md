# Learning Robust Global Representations by Penalizing Local Predictive Power
Anonymous Author(s)\*</sup>


### Requirements
- Python 3.6
- [Tensorflow]https://www.tensorflow.org/) tested on version `1.10.0`

### Installation

#### Setup virtualenv
```
conda create -n PAR_Cifar python=3.6
conda activate PAR_Cifar
pip install -r requirements.txt
```
## Pre-reqs

### Cifar10 Data
1. Download Cifar10 images:
```
cd data/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -vzxf cifar-10-python.tar.gz
```

2. Process the Cifar10 images to add perturbation:
```
python generate_data.py
```


### Model training and evaluation
Reproduce teh results of PAR and competing models. Note that for DANN we need to indicate the test domain.
```
cd experiments/
python cnn.py
python cnn_HEX.py
python cnn_InfoDrop.py
python cnn_DANN.py -test 0
python cnn_DANN.py -test 1
python cnn_DANN.py -test 2
python cnn_DANN.py -test 3
python cnn_PAR.py
```
