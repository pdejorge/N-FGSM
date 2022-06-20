import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset

class AttackUtils(object):

        def __init__(self, lower_limit, upper_limit, std):
                self.lower_limit = lower_limit
                self.upper_limit = upper_limit
                self.std = std

        def clamp(self, X, lower_limit, upper_limit):
                return torch.max(torch.min(X, upper_limit), lower_limit)

        def attack_pgd(self, model, X, y, epsilon, alpha, attack_iters, restarts):
                max_loss = torch.zeros(y.shape[0]).cuda()
                max_delta = torch.zeros_like(X).cuda()
                for zz in range(restarts):
                        delta = torch.zeros_like(X).cuda()
                        for i in range(len(epsilon)):
                                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                        delta.data = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
                        delta.requires_grad = True
                        for kk in range(attack_iters):
                                output = model(X + delta)
                                index = torch.where(output.max(1)[1] == y)
                                if len(index[0]) == 0:
                                        break
                                        
                                loss = F.cross_entropy(output, y)
                                loss.backward()
                                
                                grad = delta.grad.detach()
                                d = delta[index[0], :, :, :]
                                g = grad[index[0], :, :, :]
                                d = self.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
                                d = self.clamp(d, self.lower_limit - X[index[0], :, :, :], self.upper_limit - X[index[0], :, :, :])
                                delta.data[index[0], :, :, :] = d
                                delta.grad.zero_()
                        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
                        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                        max_loss = torch.max(max_loss, all_loss)
                return max_delta
            
        def attack_pgd_l2(self, model, X, y, epsilon, alpha, attack_iters, restarts):
                epsilon = torch.max(epsilon)
                alpha = torch.max(alpha)
                max_loss = torch.zeros(y.shape[0]).cuda()
                max_delta = torch.zeros_like(X).cuda()
                for zz in range(restarts):
                        delta = torch.zeros_like(X).cuda()
                        delta.uniform_(-epsilon, epsilon)
                        delta.data = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
                        delta.requires_grad = True
                        for kk in range(attack_iters):
                                output = model(X + delta)
                                index = torch.where(output.max(1)[1] == y)
                                if len(index[0]) == 0:
                                        break
                                loss = F.cross_entropy(output, y)
                                loss.backward()
                                grad = delta.grad.detach()
                                d = delta[index[0], :, :, :]
                                g = grad[index[0], :, :, :]
                                d = d + l2_project(g, alpha)
                                d = d + l2_clip(d, epsilon)
                                d = self.clamp(d, self.lower_limit - X[index[0], :, :, :], self.upper_limit - X[index[0], :, :, :])
                                delta.data[index[0], :, :, :] = d
                                delta.grad.zero_()
                        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
                        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                        max_loss = torch.max(max_loss, all_loss)
                return max_delta

        def attack_fgsm(self, model, X, y, epsilon, alpha):
                delta = torch.zeros_like(X).cuda()
                for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
                delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = self.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = self.clamp(delta[:X.size(0)], self.lower_limit - X, self.upper_limit - X)
                return delta

        def attack_nfgsm(self, model, X, y, epsilon=8, alpha=10, unif=1, clip=1):
                epsilon = (epsilon / 255.) / self.std
                alpha = (alpha / 255.) / self.std
                if clip > 0:
                        clip = clip * epsilon
                delta = torch.zeros_like(X).cuda()
                if unif > 0:
                        for j in range(len(epsilon)):
                                delta[:, j, :, :].uniform_(-unif * epsilon[j][0][0].item(), unif * epsilon[j][0][0].item())
                delta = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
                delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta = delta + alpha * torch.sign(grad)
                if clip != 0:
                        delta = self.clamp(delta, -epsilon, epsilon)
                delta = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
                return delta
            
        def evaluate_pgd(self, test_loader, model, attack_iters, restarts, epsilon=8, l2_norm=False):
                epsilon = (epsilon / 255.) / self.std
                alpha = (2 / 255.) / self.std
                if l2_norm:
                    alpha = epsilon / attack_iters * 2
                pgd_loss = 0
                pgd_acc = 0
                n = 0
                model.eval()
                for i, (X, y) in enumerate(test_loader):
                        X, y = X.cuda(), y.cuda()
                        if l2_norm:
                            pgd_delta = self.attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts)
                        else:
                            pgd_delta = self.attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
                        with torch.no_grad():
                                output = model(X + pgd_delta)
                                loss = F.cross_entropy(output, y)
                                pgd_loss += loss.item() * y.size(0)
                                pgd_acc += (output.max(1)[1] == y).sum().item()
                                n += y.size(0)
                return pgd_loss/n, pgd_acc/n

        def evaluate_noise(self, test_loader, model, sigma=0.1):
                test_loss = 0
                test_acc = 0
                n = 0
                model.eval()
                for i, (X, y) in enumerate(test_loader):
                        X, y = X.cuda(), y.cuda()
                        # Noise to add
                        noise = torch.normal(mean=0, std=sigma, size=X.shape).cuda()
                        # Make sure image stays within limit
                        noise = self.clamp(noise, self.lower_limit - X, self.upper_limit - X)
                        with torch.no_grad():
                                output = model(X + noise)
                                loss = F.cross_entropy(output, y)
                                test_loss += loss.item() * y.size(0)
                                test_acc += (output.max(1)[1] == y).sum().item()
                                n += y.size(0)
                return test_loss/n, test_acc/n

        def evaluate_standard(self, test_loader, model):
                test_loss = 0
                test_acc = 0
                n = 0
                model.eval()
                with torch.no_grad():
                        for i, (X, y) in enumerate(test_loader):
                                X, y = X.cuda(), y.cuda()
                                output = model(X)
                                loss = F.cross_entropy(output, y)
                                test_loss += loss.item() * y.size(0)
                                test_acc += (output.max(1)[1] == y).sum().item()
                                n += y.size(0)
                return test_loss/n, test_acc/n
            
        def evaluate_unif_fgsm(self, test_loader, model, unif=1, clip=1, alpha=10, epsilon=8):
            fgsm_loss = 0
            fgsm_acc = 0
            n = 0
            model.eval()
            for i, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                fgsm_delta = self.attack_unif_fgsm(model, X, y, epsilon, alpha, unif, clip)
                with torch.no_grad():
                    output = model(X + fgsm_delta)
                    loss = F.cross_entropy(output, y)
                    fgsm_loss += loss.item() * y.size(0)
                    fgsm_acc += (output.max(1)[1] == y).sum().item()
                    n += y.size(0)
            return fgsm_loss/n, fgsm_acc/n

