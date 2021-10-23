# Neural Architecture Search From Task Similarity Measure
This is the source code for Neural Architecture Search From Task Similarity Measure (https://arxiv.org/pdf/2103.00241.pdf)

## Description

In this paper, we propose a neural architecture search framework based on a similarity measure between some baseline tasks and a target task. We first define the notion of the task similarity based on the log-determinant of the Fisher Information matrix. Next, we compute the task similarity from each of the baseline tasks to the target task. By utilizing the relation between a target and a set of learned baseline tasks, the search space of architectures for the target task can be significantly reduced, making the discovery of the best candidates in the set of possible architectures tractable and efficient, in terms of GPU days. This method eliminates the requirement for training the networks from scratch for a given target task as well as introducing the bias in the initialization of the search space from the human domain.

## Getting Started

### Dependencies

* Requires Pytorch, Numpy
* MNIST dataset (https://www.kaggle.com/oddrationale/mnist-in-csv)
* CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)

### Executing program

* First, we define tasks in MNIST and CIFAR-10 datasets and use the CNN to train on each task. The weights of the trained CNN is saved for each task.
```
python train_task_mnist.py
python train_task_cifar_repeat.py
```
* Next, we compute the Fisher Information matrices for each pair of tasks using the base task's network. Then, we identify the closest tasks based on the log-determinant of the Fisher Information matrices
```
python log_det_distance.py
```
Lastly, the FUSE algorithm is applied to find the suitable architecture for the incoming task:
```
python NAS_FUSE.py
```

### Results
The confusion matrices below shows the mean (left) and standard deviation (right) of the distances between 8 baseline tasks from MNIST, CIFAR-10 datasets.
<p align="center">
  <img src="images/fig1.jpg" height="350" title="Mean">
  <img src="images/fig2.jpg" height="350" title="Sig">
</p>

The table below indicates the comparison of the NAS performance with handdesigned classifiers and state-of-the-art methods on Task 3 in
MNIST based on the discovered closest task, Task 7 
| Architecture | Accuracy (%) | Paramameters (M) | GPU days |
| :---         |    :---:  |     :---:        |  :---:   |
| VGG-16       | 99.55     |  14.72    | - |
| ResNet-18    | 99.56     |  11.44    | - |
| DenseNet-121 | 99.61     |  6.95     | - |
| Random Search| 99.59     |  2.23     | 4 |
| ENAS         | 97.77     |  4.60     | 4 |
| DARTS        | 99.51     |  2.37     | 2 |
| LD-NAS (ours)| 99.67     |  2.28     | 2 |

The table below indicates the comparison of the NAS performance with handdesigned classifiers and state-of-the-art methods on Task 6 in
CIFAR-10 based on the discovered closest task, Task 7. 
| Architecture | Accuracy (%) | Paramameters (M) | GPU days |
| :---         |    :---:  |     :---:        |  :---:   |
| VGG-16       | 86.75     |  14.72    | - |
| ResNet-18    | 86.93     |  11.44    | - |
| DenseNet-121 | 88.12     |  6.95     | - |
| Random Search| 88.55     |  3.65     | 5 |
| ENAS         | 75.22     |  4.60     | 4 |
| DARTS        | 90.11     |  3.12     | 2 |
| LD-NAS (ours)| 90.87     |  3.02     | 2 |

## Authors

Cat P. Le (cat.le@duke.edu), 
<br>Mohammadreza Soltani, 
<br>Robert Ravier, 
<br>Vahid Tarokh