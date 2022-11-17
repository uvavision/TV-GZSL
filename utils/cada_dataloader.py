from builtins import breakpoint
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy

import pdb

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, image_feature_type, finetuned_features, dataset='CUB', aux_datasource='attributes', device='cuda'):

        print ('dataset: ', dataset)
        print("The current working directory is")
        print(os.getcwd())
        #folder = str(Path(os.getcwd()))
        print ("New working directory")
        folder = "data/RN101/"
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        print('Project Directory:')
        print(project_directory)
        data_path = str(project_directory) # + '/data'
        print('Data Path')
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        if self.dataset == 'CUB':
            self.datadir = self.data_path + '/CUB/'
        elif self.dataset == 'SUN':
            self.datadir = self.data_path + '/SUN/'
        elif self.dataset == 'AWA1':
            self.datadir = self.data_path + '/AWA1/'
        elif self.dataset == 'AWA2':
            self.datadir = self.data_path + '/AWA2/'
            
        self.read_matdataset(image_feature_type, finetuned_features)
        self.index_in_epoch = 0
        self.epochs_completed = 0


    def next_batch(self, batch_size):
        # pcascante
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label =  self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [ batch_feature, batch_att]


    def read_matdataset(self, image_feature_type, finetuned_features):

        if image_feature_type == 'resnet101':
            path= self.datadir + 'res101.mat'
            if finetuned_features:
                path= 'data/RN101/{}/res101_finetuned.mat'.format(self.dataset)
            else:
                path= 'data/RN101/{}/res101.mat'.format(self.dataset) 
            print('_____')
            print(path)
            print ('RESNET101 FEATURES - FINETUNED:', finetuned_features)
            matcontent = sio.loadmat(path)
            # #pcascante load provided features
            feature = matcontent['features'].T
            label = matcontent['labels'].astype(int).squeeze() - 1
        else:
            BACKBONEcontent = pickle.load( open( "data/{}_{}_data.p".format(image_feature_type, self.dataset.lower()), "rb" ) )
            print ('{} FEATURES'.format(image_feature_type))
            feature = BACKBONEcontent['features']
            label = BACKBONEcontent['labels'].astype(int).squeeze() - 1

        path= self.datadir + 'att_splits.mat'
        matcontent = sio.loadmat(path)
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1


        if self.auxiliary_data_source == 'attributes':
            self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary datasource is not available for this dataset')
            else:
                with open(self.datadir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = torch.from_numpy(x[self.auxiliary_data_source]).float().to(self.device)


                print('loaded ', self.auxiliary_data_source)
        
        # breakpoint()
        # if image_feature_type in ['resnet101', 'CLIP']:
        scaler = preprocessing.MinMaxScaler()
        train_feature = scaler.fit_transform(np.concatenate([feature[trainval_loc], feature[trainval_loc]]))
        test_seen_feature = scaler.transform(np.concatenate([feature[test_seen_loc], feature[test_seen_loc]]))
        test_unseen_feature = scaler.transform(np.concatenate([feature[test_unseen_loc], feature[test_unseen_loc]]))

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(np.concatenate([label[trainval_loc], label[trainval_loc]])).long().to(self.device)
        test_unseen_label = torch.from_numpy(np.concatenate([label[test_unseen_loc], label[test_unseen_loc]])).long().to(self.device)
        test_seen_label = torch.from_numpy(np.concatenate([label[test_seen_loc], label[test_seen_loc]])).long().to(self.device)

        # breakpoint()
        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.train_mapped_label = map_label(train_label, self.seenclasses)

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]