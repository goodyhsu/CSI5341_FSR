from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .MNIST import MNIST
from .SVHN import SVHN

__all__ = ['available_datasets']
available_datasets = {'cifar10': CIFAR10 , 'cifar100': CIFAR100, 'mnist': MNIST, 'svhn': SVHN }