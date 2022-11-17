import os, errno
import numpy as np
from scipy import linalg
import random
import pickle
from itertools import repeat, cycle

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from .helpers import *

from torch.utils.data import random_split, DataLoader

from torch.autograd import Variable
import pandas as pd
import random
import pickle
from PIL import Image

from torchvision import datasets
import torchvision.transforms as transforms
from itertools import repeat, cycle
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SubsetRandomSampler


def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class CUB_2010(torch.utils.data.Dataset):
    def __init__(self, image_path, annotation_filepath, split='train', transform = None, precomputed_features=False, max_classes=0): #
        super(CUB_2010, self).__init__()

        # download cub_200_2011 from google drive
        # from google_drive_downloader import GoogleDriveDownloader as gdd
        # gdd.download_file_from_google_drive(file_id='1hbzc_P1FuxMkcabkgn9ZKinBwW683j45',
        #                                     dest_path='data/cub_200_2011.tgz',
        #                                     unzip=True)

        # from google_drive_downloader import GoogleDriveDownloader as gdd
        # gdd.download_file_from_google_drive(file_id='1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx',
        #                                     dest_path='data/images_2010.tgz',
        #                                     unzip=True)

        self.precomputed_features = precomputed_features

        print('Loading {} images list...'.format(split))
        file = open(os.path.join(annotation_filepath, '{}.txt'.format(split)), 'r')
        imgs = []
        labels = []
        imgs_features = []
        for c_lines, f in enumerate(file.readlines()):
            data = f.strip().split(' ')
            imgs.append(data[0])
            labels.append(int(data[1]))
            imgs_features.append(data[0].split('/')[-1].split('.')[0])
            if max_classes != 0 and c_lines == max_classes:
                break
        file.close()
        self.image_names = np.array(imgs)
        self.image_labels = np.array(labels)
        self.image_features_location = np.array(imgs_features)
        
        self.transform = transform
        self.image_path = image_path
                
        print('...loaded.')
    
    def __getitem__(self, index):
        image_feature_path = self.image_features_location[index]
        label = self.image_labels[index]

        if self.precomputed_features:
            img_ = pickle.load(open(os.path.join(self.image_path,  image_feature_path + '.p'), "rb" ))
            
        else:
            image_name = self.image_names[index]
            img_ = Image.open(open(os.path.join(self.image_path, image_name), 'rb'))
            img_ = img_.convert("RGB")
            if self.transform:
                img_ = self.transform(img_)

        return img_, label, self.image_names[index]
    
    def __len__(self):
        return len(self.image_names)


def load_data_subsets(dataset, data_target_dir, annotation_filepath, batch_size, test_batch_size, workers, precomputed):

    kwargs = {'num_workers': workers, 'pin_memory': True}

    if dataset in ['imagenet', 'cub2010']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(dataset)

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomChoice([transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                                    transforms.RandomGrayscale(p = 0.1)
                                            ]),
                                            transforms.RandomRotation(25),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(), transforms.Normalize(mean, std)])

    if dataset == 'imagenet':
        train_data = torchvision.datasets.ImageNet(data_target_dir, split='train', transform=train_transform)
        val_data = torchvision.datasets.ImageNet(data_target_dir, split='train', transform=test_transform)
        num_classes = 1000
    elif dataset == 'cub2010':
        train_data = CUB_2010(data_target_dir, annotation_filepath, split='train', transform=test_transform, precomputed_features=precomputed) #no train_transform
        val_data = CUB_2010(data_target_dir, annotation_filepath, split='validation', transform=test_transform, precomputed_features=precomputed) 
        test_data = CUB_2010(data_target_dir, annotation_filepath, split='test', transform=test_transform, precomputed_features=precomputed)
        num_classes = 200
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=precomputed, **kwargs)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, **kwargs)


    return num_classes, train_loader, valid_loader, test_loader

def get_train_classes_ids(annotation_filepath):
    """
    Get the seen classes ids to be used for training the classifier

    Args:
        annotation_filepath: path to .txt document containing the train split list

    Returns:
        list: seen classes ids (to use when training)
    """
    classes_ids = []
    file = open(os.path.join(annotation_filepath, 'train.txt'), 'r')
    for f in file.readlines():
        data = f.strip().split(' ')
        c_id = int(data[1]) #remember that ids start in 1, not 0!
        if c_id not in classes_ids:
            classes_ids.append(c_id)
    file.close()

    return classes_ids


class Computed_Z_Features(Dataset):
    def __init__(self, X_features, Y_targets):
        self.x = X_features
        self.y = Y_targets

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x) 


def get_training_classifier_dataloaders(workers, batch_size, train_features, train_targets, val_features, val_targets, test_features, test_targets):
    kwargs = {'num_workers': workers, 'pin_memory': True}

    train_data = Computed_Z_Features(train_features, train_targets)
    val_data = Computed_Z_Features(val_features, val_targets)
    test_data = Computed_Z_Features(test_features, test_targets)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader

