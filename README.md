# Learning Robust Global Representations by Penalizing <br>Local Predictive Power
Anonymous Author(s)\*</sup>


## Requirements
- Python 3.6 with Numpy and opencv-python
- [Tensorflow](https://www.tensorflow.org/) tested on version `1.10.0`

## Patch-wise Adversarial Regularization (PAR)
<img src="./PAR.jpg" width = "800px" />

We introduced the patch-wise adversarial regularization (PAR) that regularizes the model to focus on global concept of the depicted objects in training data by penalizing the model’s predictive power through local patches. To attest the effectiveness of our model, we verify it on a variety of domain adaptation/generalization settings, including perturbed [MNIST](http://yann.lecun.com/exdb/mnist/), perturbed [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017) and a novel large-scale domain generalization dataset ImageNet-Sketch. 

### Experiments
In general, please refer to each experiment folder for detailed codes and instruction to reproduce the benchmark mentioned in the paper. To train PAR, the following arguments are optinal and can be tuned according to the specific situation:

-e, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --epochs, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; how many epochs to run in total?<br>
-b, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --batch_size, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; batch size during training per GPU<br>
-m, &nbsp;&nbsp;&nbsp;&nbsp; --lam, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  trade-off weight of the regularization loss<br>
-adv, &nbsp;&nbsp; --adv_flag, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  adversarially training local features or not<br>
-se, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--start_epoch, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; the epoch start to adversarial training<br>
-alr, &nbsp;&nbsp;&nbsp; --adv_learning_rate, &nbsp;&nbsp;&nbsp; learning rate for adversarial learning<br>
-lr, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --learning_rate, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; initial learning rate<br>
-o, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --output, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; save model filepath<br>
-g, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --gpuid, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; which gpu to use<br>

## ImageNet-Sketch Dataset

<img src="./imagenet_sketch.jpg" width = "800px" />

Compatible with standard ImageNet validation data set for the classification task, we introduce ImageNet-Sketch dataset which consists of 50000 images, 50 images for each of the 1000 ImageNet classes. We construct the data set with Google Image queries “sketch of ”, where is the standard class name. We only search within the “black and white” color scheme. We initially query 100 images for every class, and then manually clean the pulled images by deleting the irrelevant images and images that are for similar but different classes. For some classes, there are less than 50 images after manually cleaning, then we augment the data set by flipping and rotating the images.
