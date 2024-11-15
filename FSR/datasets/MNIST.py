from .base_dataset import BaseDataset
import torchvision.transforms as transforms
import torchvision
import torch

class MNIST(BaseDataset):
    def __init__(self, args):
        self.num_classes = 10
        self.image_size = (32, 32) # To fit the model, adjust the image size to 32x32
        
        self.transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),  
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # expand to 3 channels
            # transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))

            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # expand to 3 channels
            # transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))

            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=args.bs, shuffle=True)

        self.testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=args.bs, shuffle=False)

    def get_dataset(self):
        return (self.num_classes, self.image_size, 
                self.trainloader, self.testloader, 
                self.trainset, self.testset)
