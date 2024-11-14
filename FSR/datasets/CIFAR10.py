from .base_dataset import BaseDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch

class CIFAR10(BaseDataset):
    def __init__(self, args):
        self.num_classes = 10
        self.image_size = (32, 32)
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='constant', value=0).squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=args.bs, shuffle=True)

        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=args.bs, shuffle=False)

    def get_dataset(self):
        return (self.num_classes, self.image_size, 
                self.transform_train, self.transform_test,
                self.trainloader, self.testloader, 
                self.trainset, self.testset)