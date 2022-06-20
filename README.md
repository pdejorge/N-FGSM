# N-FGSM

Official repo for the paper "Make Some Noise: Reliable and Efficient Single-Step Adversarial Training" (https://arxiv.org/abs/2202.01181)

## Requirements
You may create a conda environment from `requirements.txt`
```
conda create --name <env> --file requirements.txt
```
In general the code should be compatible with `pytorch >= 1.7.1`.

## Usage
You may perform N-FGSM Adversarial Training with CIFAR10, CIFAR100 and SVHN datasets and PreActResNet18 and WideResNet28-10. For instance, to train on CIFAR10 with preactresnet18:
```
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results --root-model-dir /path/to/trained/models/
```
The configurations for all datasets, models and epsilon radii can be found in `experiments.sh`
