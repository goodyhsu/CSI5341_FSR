import os
import urllib.request
import zipfile
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .base_dataset import BaseDataset

class TimgNet(BaseDataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform

        
        self.train = train

        if self.train:
            self.data = []
            self.labels = []
            classes = os.listdir(os.path.join(root_dir, 'train'))
            for label, cls in enumerate(classes):
                cls_dir = os.path.join(root_dir, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    self.data.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)
        else:
            self.data = []
            self.labels = []
            test_dir = os.path.join(root_dir, 'test', 'images')
            test_annotations = pd.read_csv(os.path.join(root_dir, 'test', 'test_annotations.txt'), 
                                          sep='\t', header=None, 
                                          names=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'])
            class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(os.path.join
                                                                                 (root_dir, 'train'))))}
            for _, row in test_annotations.iterrows():
                self.data.append(os.path.join(test_dir, row['file_name']))
                self.labels.append(class_to_idx[row['class']])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


    def download_dataset():
        print('Beginning dataset download with urllib2')
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        path = "%s/tiny-imagenet-200.zip" % os.getcwd()
        urllib.request.urlretrieve(url, path)
        print("Dataset downloaded")

    def unzip_data():
        path_to_zip_file = "%s/tiny-imagenet-200.zip" % os.getcwd()
        directory_to_extract_to = os.getcwd()
        print("Extracting zip file: %s" % path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        print("Extracted at: %s" % directory_to_extract_to)


    def get_dataset(self):
        self.download_dataset()
        self.unzip_data()
        data_dir = './tiny-imagenet-200'

        self.transform_train = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
        self.transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
        self.num_classes = 200
        self.image_size = (32,32)

        self.trainset = self.TinyImageNetDataset(root_dir=data_dir, transform=self.transform_train, train=True)
        self.testset = self.TinyImageNetDataset(root_dir=data_dir, transform=self.transform_test, train=False)

        self.trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=32, shuffle=False)

        return (self.num_classes, self.image_size, 
                self.transform_train, self.transform_test,
                self.trainloader, self.testloader, 
                self.trainset, self.testset)