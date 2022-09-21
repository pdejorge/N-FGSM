import argparse
import copy
import logging
import os
import time

import numpy as np
from scipy.stats import ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.preact_resnet import PreActResNet18
from architectures.wide_resnet import Wide_ResNet

from utils.data_utils import CIFAR10Utils, SVHNUtils, CIFAR100Utils
from utils.attack_utils import AttackUtils

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    
    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100 or SVHN')
    parser.add_argument('--architecture', default='PreActResNet18', type=str,
                        help='One of: wideresnet, preactresnet18. Default: preactresnet18.')
    parser.add_argument('--wide_resnet_depth', default=28, type=int, help='WideResNet depth')
    parser.add_argument('--wide_resnet_width', default=10, type=int, help='WideResNet width')
    parser.add_argument('--wide_resnet_dropout_rate', default=0.3, type=float, help='WideResNet dropout rate')
    
    # Training schedule settings
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    
    
    # Adversarial training and evaluation settings
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--epsilon_test', default=None, type=int, 
                        help='''Epsilon to be used at test time (only for final model,
                        if computing loss during training epsilon train is used).
                        If set to None, default, the same args.epsilon will be used for test and train.''')
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--unif', default='2.0', type=float,
                        help='''Magnitude of the uniform noise relative to epsilon.
                                - k -> U(-k*eps, k*eps),
                                - 0 -> No noise,
                                - Default is 2 -> U(-2eps, 2eps).
                        ''')
    parser.add_argument('--clip', default=-1, type=float,
                        help='''Radius of the inf ball where to clip the perturbations.
                                Relative to epsilon: i.e. 1 means clip(-eps, eps).
                                By default it is set to -1 (no clipping)
                                In Fast Adv Training it would be set to 1.
                        ''')
    parser.add_argument('--robust_test_size', default=-1, type=int,
                        help='Number of samples to be used for robust testing, Default: -1 will use all samples')
    parser.add_argument('--validation-early-stop', action='store_true',
        help='Store best epoch via validation')

    
    # Config paths
    parser.add_argument('--out-dir', default='/path/to/results/',
                        type=str, help='Output directory')
    parser.add_argument('--root-model-dir',
                        default='/path/to/trained/models/',
                        type=str, help='Models directory')
    
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    
    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    print('Defining data object')
    if args.dataset.upper() == 'CIFAR10':
        data_utils = CIFAR10Utils()
    elif args.dataset.upper() == 'SVHN':
        data_utils = SVHNUtils()
        args.lr_max = 0.05
    elif args.dataset.upper() == 'CIFAR100':
        data_utils = CIFAR100Utils()
    else:
        raise ValueError('Unsupported dataset.')
        
    # If args.epsilon_test is None, use the same epsilon than during training.
    if args.epsilon_test is None:
        args.epsilon_test = args.epsilon

    print('Defining attack object')
    attack_utils = AttackUtils(data_utils.lower_limit, data_utils.upper_limit, data_utils.std)

    # Set-up results paths
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    model_path = os.path.join(args.root_model_dir, args.out_dir.split('/')[-1])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.validation_early_stop:
        best_pgd_val_acc = 0
        valid_size = 1000
    else:
        valid_size = 0

    (train_loader, test_loader, robust_test_loader,
                valid_loader, train_idx, valid_idx) = data_utils.get_indexed_loaders(args.data_dir,
                                                                        args.batch_size,
                                                                        valid_size=valid_size,
                                                                        robust_test_size=args.robust_test_size)

    # Making sure that data is in the supported format
    if (data_utils.img_size != (32,32)):
        raise RuntimeError('Data is not in the supported format input image size (32x32)')

    # Adv training and test settings
    epsilon = (args.epsilon / 255.) / data_utils.std
    alpha = (args.alpha / 255.) / data_utils.std
    pgd_alpha = (2 / 255.) / data_utils.std
    if args.clip > 0:
        clip = (args.clip * args.epsilon / 255.) / data_utils.std
        
    # For SVHN, we increase the perturbation radius from 0 to epsilon for first 5 epochs
    if args.dataset.upper() == 'SVHN':
        epsilon_global = epsilon
        alpha_global = alpha
        if args.clip > 0:
            clip_global = clip
        n_warmup_epochs = 5
        n_warmup_iterations = n_warmup_epochs * len(train_loader)
        
    # Define architecture
    args.num_classes = data_utils.max_label + 1 # Labels start from 0
    if args.architecture.upper() == 'PREACTRESNET18':
        model = PreActResNet18(num_classes=args.num_classes).cuda()
    elif args.architecture.upper() in  'WIDERESNET':
        logger.info(f'Using WideResNet with depth {args.wide_resnet_depth},')
        logger.info(f'width {args.wide_resnet_width} and Dropout rate {args.wide_resnet_dropout_rate}')
        model = Wide_ResNet(args.wide_resnet_depth,
                            args.wide_resnet_width,
                            args.wide_resnet_dropout_rate,
                            num_classes=args.num_classes).cuda()

    else:
        raise ValueError('Unknown architecture.')

    model.train()
    
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        if args.dataset.upper() == 'SVHN':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps * 2 / 5, step_size_down=lr_steps * 3 / 5)
        else:
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    prev_robust_acc = 0.
    start_train_time = time.time()
    if args.validation_early_stop:
        val_acc_hist = []
        robust_val_acc_hist = []

    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    iter_count = 0

    max_iter = len(train_loader) * args.epochs
    counter_iter = 0
    
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y, batch_idx) in enumerate(train_loader):
            # For SVHN, we increase the perturbation radius from 0 to epsilon for first 5 epochs
            if args.dataset.upper() == 'SVHN':
                epsilon = epsilon_global * min(iter_count / n_warmup_iterations, 1)
                alpha = alpha_global * min(iter_count / n_warmup_iterations, 1)
                if args.clip > 0:
                    clip = clip_global * min(iter_count / n_warmup_iterations, 1)
                
            batch_size_ = X.shape[0]
            X, y = X.cuda(), y.cuda()
            
            # Initialize random step
            eta = torch.zeros_like(X).cuda()
            if args.unif > 0:
                for j in range(len(epsilon)):
                    eta[:, j, :, :].uniform_(-args.unif * epsilon[j][0][0].item(), args.unif * epsilon[j][0][0].item())
                eta = attack_utils.clamp(eta, attack_utils.lower_limit - X, attack_utils.upper_limit - X)
            eta.requires_grad = True

            output = model(X + eta)
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, eta)[0]
            grad = grad.detach()
            # Compute perturbation based on sign of gradient
            delta = eta + alpha * torch.sign(grad)

            delta = attack_utils.clamp(delta, attack_utils.lower_limit - X, attack_utils.upper_limit - X)
            if args.clip > 0:
                delta = attack_utils.clamp(delta, -clip, clip)

            delta = delta.detach()
            
            output = model(X + delta)

            # Training step
            loss = F.cross_entropy(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

            iter_count += 1

        if args.validation_early_stop:
            pgd_loss, pgd_acc = attack_utils.evaluate_pgd(valid_loader, model, 10, 1, epsilon=args.epsilon)
            test_loss, test_acc = attack_utils.evaluate_standard(valid_loader, model)
            # After evaluating the model, set model to train mode again
            model.train()
            if pgd_acc >= best_pgd_val_acc:
                best_pgd_val_acc = pgd_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                val_acc_hist.append(test_acc)
                robust_val_acc_hist.append(pgd_acc)

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)

    train_time = time.time()
    if args.validation_early_stop:
        torch.save(best_state_dict, os.path.join(model_path, f'best_model.pth'))

    final_state_dict = model.state_dict()
    torch.save(final_state_dict, os.path.join(model_path, 'model.pth'))

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    if args.validation_early_stop:
        np.save(os.path.join(model_path, 'val_acc_hist.npy'), val_acc_hist)
        np.save(os.path.join(model_path, 'robust_val_acc.npy'), robust_val_acc_hist)


    if args.robust_test_size != 0:
        # Evaluation final model
        print('Training finished, starting evaluation')
        args.num_classes = data_utils.max_label + 1 # Labels start from 0
        if args.architecture.upper() == 'PREACTRESNET18':
            model_test = PreActResNet18(num_classes=args.num_classes).cuda()
        elif args.architecture.upper() in  'WIDERESNET':
            logger.info(f'Using WideResNet with depth {args.wide_resnet_depth},')
            logger.info(f'width {args.wide_resnet_width} and Dropout rate {args.wide_resnet_dropout_rate}')
            model_test = Wide_ResNet(args.wide_resnet_depth,
                                args.wide_resnet_width,
                                args.wide_resnet_dropout_rate,
                                num_classes=args.num_classes).cuda()
            
        model_test.load_state_dict(final_state_dict)
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = attack_utils.evaluate_pgd(robust_test_loader, model_test, 50, 10, epsilon=args.epsilon_test)
        test_loss, test_acc = attack_utils.evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        print('Finished evaluating final model')

        if args.validation_early_stop:
            # Evaluation best model
            print('Evaluating best model')
            args.num_classes = data_utils.max_label + 1 # Labels start from 0
            print('Training finished, starting evaluation')
        
            args.num_classes = data_utils.max_label + 1 # Labels start from 0
            if args.architecture.upper() == 'PREACTRESNET18':
                best_model = PreActResNet18(num_classes=args.num_classes).cuda()
            elif args.architecture.upper() in  'WIDERESNET':
                logger.info(f'Using WideResNet with depth {args.wide_resnet_depth},')
                logger.info(f'width {args.wide_resnet_width} and Dropout rate {args.wide_resnet_dropout_rate}')
                best_model = Wide_ResNet(args.wide_resnet_depth,
                                    args.wide_resnet_width,
                                    args.wide_resnet_dropout_rate,
                                    num_classes=args.num_classes).cuda()

            best_model.load_state_dict(best_state_dict)
            best_model.float()
            best_model.eval()
            
            best_pgd_loss, best_pgd_acc = attack_utils.evaluate_pgd(robust_test_loader, best_model, 50, 10, epsilon=args.epsilon_test)
            best_test_loss, best_test_acc = attack_utils.evaluate_standard(test_loader, best_model)

            logger.info('Best test Loss \t Best test Acc \t Best PGD Loss \t Best PGD Acc')
            logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', best_test_loss, best_test_acc, best_pgd_loss, best_pgd_acc)

if __name__ == "__main__":
    main()
