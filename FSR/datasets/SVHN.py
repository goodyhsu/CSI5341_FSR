from .base_dataset import BaseDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch

class SVHN(BaseDataset):
    def __init__(self, args):
        self.num_classes = 10
        self.image_size = (32, 32)
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),])
    
        self.transform_test = transforms.Compose([
        transforms.ToTensor(),])

        self.trainset = torchvision.datasets.SVHN(
            root='./data', split='train', download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=args.bs, shuffle=True)

        self.testset = torchvision.datasets.SVHN(
            root='./data', split='test', download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=args.bs, shuffle=False)
        
    def get_dataset(self):
        return (self.num_classes, self.image_size, 
                self.trainloader, self.testloader, 
                self.trainset, self.testset)