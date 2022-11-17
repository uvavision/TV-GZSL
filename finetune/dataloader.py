import os
from PIL import Image
from collections import OrderedDict
import sys
from pathlib import Path
import pickle
import copy
import glob
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class DATA(Dataset):
    def __init__(self, loader_type='trainval_loc', transform=None, dataset='CUB'):

        datadir = f'../../data/RN101/{dataset}/'
        path = datadir + 'res101.mat'
        print('_____')
        print(path)
        matcontent = sio.loadmat(path)
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent.keys()

        datadir = f'../../data/RN101/{dataset}/'
        path = datadir + 'att_splits.mat'
        print('_____')
        print(path)
        attributes_mat = sio.loadmat(path)
        label = matcontent['labels'].astype(int).squeeze() - 1
        trainval_loc = attributes_mat['trainval_loc'].squeeze() - 1
        test_unseen_loc = attributes_mat['test_unseen_loc'].squeeze() - 1
        test_seen_loc = attributes_mat['test_seen_loc'].squeeze() - 1
        train_label = label[trainval_loc]
        test_unseen_label = label[test_unseen_loc]
        test_seen_label = label[test_seen_loc]
        print ('\nSeen classes:', np.unique(test_seen_label))
        print ('\nUnseen classes:', np.unique(test_unseen_label))

        if dataset == "CUB":
            paths_from_files = [__f[0][0].replace('/BS/Deep_Fragments/work/MSc/CUB_200_2011/CUB_200_2011/images/', '') for __f in matcontent['image_files']]
            labels_dict = {int(file.split('/')[0].split('.')[0]):file.split('/')[0].split('.')[1] for file in paths_from_files}
            data_path = '../data/CUB_200_2011/images/'
            imagenet_classes = [val.replace('_', ' ') for key,val in labels_dict.items()]
            text_descriptions = [ 'Image of a {}'.format(class_tmp) for class_tmp in imagenet_classes ]
            self.text_descriptions_classes = text_descriptions

            attributes_labels_filepath = '../data/CUB_200_2011/attributes.txt'
            attributes_text = []
            with open(attributes_labels_filepath) as f:
                attributes_text = f.readlines()
            attributes_text = [att.split(' ')[1].replace('\n', '').replace('_', ' ').replace('has', '')[1:] for att in attributes_text]
            attributes__text_forclip = []
            for att_tmp in attributes_text:
            #     att_tmp = att.split(' ')[1].replace('\n', '').replace('_', ' ').replace('has', '')
                attributes__text_forclip.append('with a ' + att_tmp.split('::')[1] + " " + att_tmp.split('::')[0])
            text_descriptions = [ 'Image of a bird {}'.format(class_tmp).replace('color', '').replace('with a', 'with').strip() for class_tmp in attributes__text_forclip ]
            self.text_descriptions_atts = text_descriptions

        elif dataset == "SUN":
            paths_from_files = [__f[0][0].replace('/BS/Deep_Fragments/work/MSc/data/SUN/images/', '') for __f in matcontent['image_files']]
            labels_dict = {(file.split('/')[1]):file.split('/')[2] for file in paths_from_files}
            data_path = '../data/SUN/images/'
            imagenet_classes = [cls_tmp[0][0].replace('_', ' ') for cls_tmp in attributes_mat['allclasses_names']]
            text_descriptions = [ 'Image of a {}'.format(class_tmp) for class_tmp in imagenet_classes ]
            self.text_descriptions_classes = text_descriptions

            attribute_words_tmp = sio.loadmat('/vislang/paola/bigtemp/pc9za/SUN/SUNAttributeDB/attributes.mat')
            attribute_words = [att_tmp[0][0] for att_tmp in attribute_words_tmp['attributes']]
            text_descriptions = []
            for idx, att_tmp in enumerate(attribute_words):
                if 'ing' in att_tmp:
                        text_descriptions.append('Image of a place for {}'.format(att_tmp))
                else:
                    text_descriptions.append('Image of a place with {}'.format(att_tmp))
            self.text_descriptions_atts = text_descriptions

        elif dataset == "AWA2":
            paths_from_files = [__f[0][0].replace('/BS/xian/work/data/Animals_with_Attributes2//JPEGImages/', '') for __f in matcontent['image_files']]
            labels_dict = {(file.split('/')[0]):file.split('/')[1] for file in paths_from_files}
            data_path = '../data/AWA2/Animals_with_Attributes2/JPEGImages/'
            imagenet_classes = [cls_tmp[0][0].replace('+', ' ') for cls_tmp in attributes_mat['allclasses_names']]
            text_descriptions = [ 'Image of a {}'.format(class_tmp) for class_tmp in imagenet_classes ]
            self.text_descriptions_classes = text_descriptions

            attributes_labels_filepath = '..data/AWA2/Animals_with_Attributes2/predicates.txt'
            attributes_text = []
            with open(attributes_labels_filepath) as f:
                attributes_text = f.readlines()
            attributes_text = [att.replace('     ', '').split('\t')[-1].strip() for att in attributes_text]
            text_descriptions = []
            for class_tmp in attributes_text:
                if class_tmp in ['patches', 'spots ', 'stripes ', 'flippers ', 'hands ', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail', 
                                'teeth', 'horns', 'claws', 'tusks', 'muscle', 'agility']:
                    text_descriptions.append('Image of an animal with {}'.format(class_tmp))
                elif class_tmp in ['flys', 'hops', 'swims', 'tunnels', 'walks', 'hibernate']:
                    text_descriptions.append('Image of an animal that {}'.format(class_tmp))
                elif class_tmp in ['desert', 'bush', 'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean', 'ground', 'water', 'tree', 
                                'cave',]:
                    text_descriptions.append('Image of an animal that lives in the {}'.format(class_tmp))
                else:
                    text_descriptions.append('Image of a {} animal'.format(class_tmp))
            self.text_descriptions_atts = text_descriptions


        all_atts_sentences = {}
        for idx, atts_tmp in enumerate(attributes_mat['original_att'].T):
            keyname_tmp = attributes_mat['allclasses_names'][idx][0][0].replace('_', ' ')
            all_atts_sentences[keyname_tmp] = []
            for idx_atts, att_val in enumerate(atts_tmp):
                if att_val > 40:
                    all_atts_sentences[keyname_tmp].append(self.text_descriptions_atts[idx_atts])
        self.all_atts_sentences = all_atts_sentences
            

        indices = attributes_mat[loader_type].squeeze() - 1
        self.imgs_path = data_path # 
        self.file_list = np.asarray(paths_from_files)[indices]
        self.transform = transform
        self.indices = indices
        self.matcontent = matcontent
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.imgs_path + self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        label = self.matcontent['labels'][self.indices][idx].astype(np.int16) - 1 

        desc = random.choice(self.all_atts_sentences[self.attributes_mat['allclasses_names'][label][0][0].replace('_', ' ')])
        class_desc = self.text_descriptions_classes[label]
        
        return img, label, self.file_list[idx], desc, class_desc
