import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset


class CIFAR10Utils(object):

        def __init__(self, normalize=True):

                if normalize: 
                    self.cifar10_mean = (0.4914, 0.4822, 0.4465)
                    self.cifar10_std = (0.2471, 0.2435, 0.2616)
                else:
                    self.cifar10_mean = (0., 0., 0.)
                    self.cifar10_std = (1., 1., 1.)

                self.mu = torch.tensor(self.cifar10_mean).view(3,1,1).cuda()
                self.std = torch.tensor(self.cifar10_std).view(3,1,1).cuda()

                self.upper_limit = ((1 - self.mu)/ self.std)
                self.lower_limit = ((0 - self.mu)/ self.std)


        def get_indexed_loaders(self, dir_, batch_size, batch_size_test=None, valid_size=1000, shuffle=True, robust_test_size=-1):

                if batch_size_test is None:
                        batch_size_test = batch_size

                train_dataset = IndexedCIFAR10Dataset(self.cifar10_mean, self.cifar10_std, dir_, train=True, download=True)

                # storing this data to check on main
                self.img_size = (train_dataset.dataset.data.shape[1],train_dataset.dataset.data.shape[2])
                self.max_label = np.max(train_dataset.dataset.targets)

                test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(self.cifar10_mean, self.cifar10_std),
                ])
                valid_dataset = datasets.CIFAR10(
                        dir_, train=True, transform=test_transform, download=True)
                test_dataset = datasets.CIFAR10(
                        dir_, train=False, transform=test_transform, download=True)

                num_workers = 0
                num_train = len(train_dataset)
                indices = list(range(num_train))
                if shuffle:
                        np.random.shuffle(indices)

                train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
                train_sampler = SubsetRandomSampler(train_idx)
                if valid_size > 0:
                        valid_sampler = SubsetRandomSampler(valid_idx)
                        
                # Selecting samples for robust test acc evaluation
                assert robust_test_size <= len(test_dataset)
                if robust_test_size < 0:
                    robust_test_size = len(test_dataset)
                robust_test_sampler = SubsetRandomSampler(list(range(robust_test_size)))

                train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers,
                        sampler=train_sampler
                )
                if valid_size > 0:
                        valid_loader = torch.utils.data.DataLoader(
                                dataset=valid_dataset,
                                batch_size=batch_size_test,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=valid_sampler
                        )
                else:
                        valid_loader = None
                test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers
                )
                robust_test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers,
                        sampler=robust_test_sampler
                )

                return train_loader, test_loader, robust_test_loader, valid_loader, train_idx, valid_idx


class IndexedCIFAR10Dataset(Dataset):
        def __init__(self, mean, std, root='../../data', download=False, train=True):
                train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                ])
                self.dataset = datasets.CIFAR10(root=root,
                                                download=download,
                                                train=train,
                                                transform=train_transform)

        def __getitem__(self, index):
                data, target = self.dataset[index]
                return data, target, index

        def __len__(self):
                return len(self.dataset)


class SVHNUtils(object):

        def __init__(self, normalize=True):
                # NOTE found them here --> https://www.programcreek.com/python/example/105105/torchvision.datasets.SVHN
                # TODO re-compute them from data
                if normalize:
                    self.svhn_mean = (0.4380, 0.4440, 0.4730)
                    self.svhn_std = (0.1751, 0.1771, 0.1744)
                else:
                    self.svhn_mean = (0., 0., 0.)
                    self.svhn_std = (1., 1., 1.)
                    
                self.max_label = 9
                self.img_size = (32, 32)

                self.mu = torch.tensor(self.svhn_mean).view(3,1,1).cuda()
                self.std = torch.tensor(self.svhn_std).view(3,1,1).cuda()

                self.upper_limit = ((1 - self.mu)/ self.std)
                self.lower_limit = ((0 - self.mu)/ self.std)
                

        def get_indexed_loaders(self, dir_, batch_size, batch_size_test=None, valid_size=1000, shuffle=True, robust_test_size=-1):

                if batch_size_test is None:
                        batch_size_test = batch_size

                train_dataset = IndexedSVHNDataset(self.svhn_mean, self.svhn_std, dir_, train=True, download=True)


                test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(self.svhn_mean, self.svhn_std),
                ])
                valid_dataset = datasets.SVHN(
                        dir_, split='train', transform=test_transform, download=True)
                test_dataset = datasets.SVHN(
                        dir_, split='test', transform=test_transform, download=True)

                num_workers = 0
                num_train = len(train_dataset)
                indices = list(range(num_train))
                if shuffle:
                        np.random.shuffle(indices)

                train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
                train_sampler = SubsetRandomSampler(train_idx)
                if valid_size > 0:
                        valid_sampler = SubsetRandomSampler(valid_idx)
                        
                # Selecting samples for robust test acc evaluation
                assert robust_test_size <= len(test_dataset)
                if robust_test_size < 0:
                    robust_test_size = len(test_dataset)
                robust_test_sampler = SubsetRandomSampler(list(range(robust_test_size)))

                train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers,
                        sampler=train_sampler
                )
                if valid_size > 0:
                        valid_loader = torch.utils.data.DataLoader(
                                dataset=valid_dataset,
                                batch_size=batch_size_test,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=valid_sampler
                        )
                else:
                        valid_loader = None
                test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers
                )
                robust_test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers,
                        sampler=robust_test_sampler
                )

                return train_loader, test_loader, robust_test_loader, valid_loader, train_idx, valid_idx



class IndexedSVHNDataset(Dataset):
        def __init__(self, mean, std, root='../../data', download=False, train=True):
                train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                ])
                self.dataset = datasets.SVHN(root=root,
                                             download=download,
                                             split='test' if not train else 'train',
                                             transform=train_transform)

        def __getitem__(self, index):
                data, target = self.dataset[index]
                return data, target, index

        def __len__(self):
                return len(self.dataset)
            
            
class CIFAR100Utils(object):

        def __init__(self, normalize=True):

                if normalize:
                    self.dset_mean = (0.5071, 0.4865, 0.4409)
                    self.dset_std = (0.2673, 0.2564, 0.2762)
                else:
                    self.dset_mean = (0., 0., 0.)
                    self.dset_std = (1., 1., 1.)

                self.mu = torch.tensor(self.dset_mean).view(3,1,1).cuda()
                self.std = torch.tensor(self.dset_std).view(3,1,1).cuda()

                self.upper_limit = ((1 - self.mu)/ self.std)
                self.lower_limit = ((0 - self.mu)/ self.std)

        def get_indexed_loaders(self, dir_, batch_size, batch_size_test=None, valid_size=1000, shuffle=True, robust_test_size=-1):

                if batch_size_test is None:
                        batch_size_test = batch_size

                train_dataset = IndexedCIFAR100Dataset(self.dset_mean, self.dset_std, dir_, train=True, download=True)

                # storing this data to check on main
                self.img_size = (train_dataset.dataset.data.shape[1],train_dataset.dataset.data.shape[2])
                self.max_label = np.max(train_dataset.dataset.targets)

                test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(self.dset_mean, self.dset_std),
                ])
                valid_dataset = datasets.CIFAR100(
                        dir_, train=True, transform=test_transform, download=True)
                test_dataset = datasets.CIFAR100(
                        dir_, train=False, transform=test_transform, download=True)

                num_workers = 0
                num_train = len(train_dataset)
                indices = list(range(num_train))
                if shuffle:
                        np.random.shuffle(indices)

                train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
                train_sampler = SubsetRandomSampler(train_idx)
                if valid_size > 0:
                        valid_sampler = SubsetRandomSampler(valid_idx)
                        
                # Selecting samples for robust test acc evaluation
                assert robust_test_size <= len(test_dataset)
                if robust_test_size < 0:
                    robust_test_size = len(test_dataset)
                robust_test_sampler = SubsetRandomSampler(list(range(robust_test_size)))

                train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers,
                        sampler=train_sampler
                )
                if valid_size > 0:
                        valid_loader = torch.utils.data.DataLoader(
                                dataset=valid_dataset,
                                batch_size=batch_size_test,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                sampler=valid_sampler
                        )
                else:
                        valid_loader = None
                test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers
                )
                robust_test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_workers,
                        sampler=robust_test_sampler
                )

                return train_loader, test_loader, robust_test_loader, valid_loader, train_idx, valid_idx


class IndexedCIFAR100Dataset(Dataset):
        def __init__(self, mean, std, root='../../data', download=False, train=True):
                train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                ])
                self.dataset = datasets.CIFAR100(root=root,
                                                download=download,
                                                train=train,
                                                transform=train_transform)

        def __getitem__(self, index):
                data, target = self.dataset[index]
                return data, target, index

        def __len__(self):
                return len(self.dataset)

