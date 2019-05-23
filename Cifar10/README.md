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
cd ..
mkdir cachedir/
mkdir cachedir/logs
mkdir cachedir/models
python cnn.py > cachedir/logs/cnn.txt
python cnn_HEX.py > cachedir/logs/cnn_HEX.txt
python cnn_InfoDrop.py > cachedir/logs/cnn_InfoDrop.txt
python cnn_DANN.py -test 0 > cachedir/logs/cnn_DANN_gs.txt
python cnn_DANN.py -test 1 > cachedir/logs/cnn_DANN_ng.txt
python cnn_DANN.py -test 2 > cachedir/logs/cnn_DANN_ro.txt
python cnn_DANN.py -test 3 > cachedir/logs/cnn_DANN_rm.txt
python cnn_PAR.py > cachedir/logs/cnn_PAR.txt
```
