# Commands to launch N-FGSM experiments (Ignored options will be set to default. See train.py)

## PREACTRESNET Architecture

### CIFAR 10
python train.py --epsilon 2 --alpha 2 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 4 --alpha 4 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 6 --alpha 6 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 10 --alpha 10 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 12 --alpha 12 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 14 --alpha 14 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 16 --alpha 16 --unif 2 --dataset CIFAR10 --architecture preactresnet18 --out-dir path/to/results
   
### CIFAR 100
python train.py --epsilon 2 --alpha 2 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 4 --alpha 4 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 6 --alpha 6 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 10 --alpha 10 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 12 --alpha 12 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 14 --alpha 14 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 16 --alpha 16 --unif 2 --dataset CIFAR100 --architecture preactresnet18 --out-dir path/to/results
    
### SVHN
python train.py --epsilon 2 --alpha 2 --unif 2 --dataset SVHN --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 4 --alpha 4 --unif 2 --dataset SVHN --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 6 --alpha 6 --unif 2 --dataset SVHN --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset SVHN --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 10 --alpha 10 --unif 2 --dataset SVHN --architecture preactresnet18 --out-dir path/to/results
python train.py --epsilon 12 --alpha 12 --unif 3 --dataset SVHN --architecture preactresnet18 --out-dir path/to/results
    
    
## WIDERESNET Architecture

### CIFAR 10
python train.py --epsilon 2 --alpha 2 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 4 --alpha 4 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 6 --alpha 6 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 10 --alpha 10 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 12 --alpha 12 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 14 --alpha 14 --unif 2 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 16 --alpha 16 --unif 4 --dataset CIFAR10 --architecture wideresnet --out-dir path/to/results
  
### CIFAR 100
python train.py --epsilon 2 --alpha 2 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 4 --alpha 4 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 6 --alpha 6 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 10 --alpha 10 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 12 --alpha 12 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 14 --alpha 14 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 16 --alpha 16 --unif 2 --dataset CIFAR100 --architecture wideresnet --out-dir path/to/results
  
### SVHN
python train.py --epsilon 2 --alpha 2 --unif 2 --dataset SVHN --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 4 --alpha 4 --unif 2 --dataset SVHN --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 6 --alpha 6 --unif 2 --dataset SVHN --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 8 --alpha 8 --unif 2 --dataset SVHN --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 10 --alpha 8 --unif 2 --dataset SVHN --architecture wideresnet --out-dir path/to/results
python train.py --epsilon 12 --alpha 8 --unif 2 --dataset SVHN --architecture wideresnet --out-dir path/to/results
    

