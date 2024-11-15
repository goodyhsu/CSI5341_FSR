import os
import numpy as np
import urllib.request
import zipfile
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from .base_dataset import BaseDataset

class TimgNet(BaseDataset):
    def __init__(self, args):
        self.root_dir = './data/'
        self.data_dir = './data/tiny-imagenet-200'
        
        if not os.path.exists(self.data_dir):
            self.download_dataset()
            self.unzip_data()
        
    class Split(Dataset):  
        def __init__(self, data_dir, transform=None, train=True):
            self.data = []
            self.labels = []
            self.transform = transform
            
            if train:
                classes = os.listdir(os.path.join(data_dir, 'train'))
                for label, cls in enumerate(classes):
                    cls_dir = os.path.join(data_dir, 'train', cls, 'images')
                    for img_name in os.listdir(cls_dir):
                        self.data.append(os.path.join(cls_dir, img_name))
                        self.labels.append(label)
            else:
                test_dir = os.path.join(data_dir, 'val', 'images')
                test_annotations = pd.read_csv(os.path.join(data_dir, 'val', 'val_annotations.txt'), 
                                            sep='\t', header=None, 
                                            names=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'])
                class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(os.path.join
                                                                                    (data_dir, 'train'))))}
                for _, row in test_annotations.iterrows():
                    self.data.append(os.path.join(test_dir, row['file_name']))
                    self.labels.append(class_to_idx[row['class']])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img_path = self.data[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            
            return image, label


    def download_dataset(self):
        print('Beginning dataset download with urllib2')
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        path = f"{self.root_dir}/tiny-imagenet-200.zip"
        urllib.request.urlretrieve(url, path)
        print("Dataset downloaded")

    def unzip_data(self):
        path_to_zip_file = f"{self.root_dir}/tiny-imagenet-200.zip"
        directory_to_extract_to = self.root_dir
        print("Extracting zip file: %s" % path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        print("Extracted at: %s" % directory_to_extract_to)
        
    def get_dataset(self):
        
        self.transform_train = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
        self.transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
        self.num_classes = 200
        self.image_size = (32,32)

        self.trainset = self.Split(data_dir=self.data_dir, transform=self.transform_train, train=True)
        self.testset = self.Split(data_dir=self.data_dir, transform=self.transform_test, train=False)
        self.trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=32, shuffle=False)

        return (self.num_classes, self.image_size, 
                self.trainloader, self.testloader, 
                self.trainset, self.testset)