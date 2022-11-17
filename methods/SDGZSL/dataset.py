from builtins import breakpoint
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import pickle

class DATA_LOADER(object):
    def __init__(self, opt):
        self.finetune = opt.finetune
        if opt.dataset in ['FLO_EPGN','CUB_STC']:
            if self.finetune:
                self.read_fine_tune(opt)
            else:
                self.read(opt)
        elif opt.dataset in ['CUB', 'AWA2', 'APY', 'FLO', 'SUN']:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_fine_tune(self,opt):
        if opt.dataset == "CUB_STC":
            opt.dataset = "CUB"
        if opt.dataset == "FLO_EPGN":
            opt.dataset = "FLO"

        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/cub_feat.mat")
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + "_finetuned.mat")

        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        # feature = matcontent['features'].T
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        train_att = matcontent['att_train']
        seen_pro = matcontent['seen_pro']
        attribute = matcontent['attribute']
        unseen_pro = matcontent['unseen_pro']
        self.attribute = torch.from_numpy(attribute).float()
        self.train_att = seen_pro.astype(np.float32)
        self.test_att = unseen_pro.astype(np.float32)

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()

    def read(self, opt):
        if opt.dataset == "CUB_STC":
            opt.dataset = "CUB"
        if opt.dataset == "FLO_EPGN":
            opt.dataset = "FLO"

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        train_att = matcontent['att_train']
        seen_pro = matcontent['seen_pro']
        attribute = matcontent['attribute']
        unseen_pro = matcontent['unseen_pro']
        self.attribute = torch.from_numpy(attribute).float()
        self.train_att = seen_pro.astype(np.float32)
        self.test_att = unseen_pro.astype(np.float32)

        train_fea = matcontent['train_fea']
        test_seen_fea = matcontent['test_seen_fea']
        test_unseen_fea = matcontent['test_unseen_fea']

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(train_fea)
        _test_seen_feature = scaler.transform(test_seen_fea)
        _test_unseen_feature = scaler.transform(test_unseen_fea)
        mx = _train_feature.max()
        train_fea = train_fea * (1 / mx)
        test_seen_fea = test_seen_fea * (1 / mx)
        test_unseen_fea = test_unseen_fea * (1 / mx)

        self.train_feature = torch.from_numpy(train_fea).float()
        self.test_seen_feature = torch.from_numpy(test_seen_fea).float()
        self.test_unseen_feature = torch.from_numpy(test_unseen_fea).float()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/label.mat")

        train_idx = matcontent['train_idx'] - 1
        train_label = matcontent['train_label_new']
        test_unseen_idex = matcontent['test_unseen_idex'] - 1
        test_seen_idex = matcontent['test_seen_idex'] - 1
        self.train_label = torch.from_numpy(train_idx.squeeze()).long()
        self.test_seen_label = torch.from_numpy(test_seen_idex.squeeze()).long()
        self.test_unseen_label = torch.from_numpy(test_unseen_idex.squeeze()).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

    def read_matdataset(self, opt):

        # breakpoint()
        if opt.feature_backbone == 'resnet101':
            matcontent = sio.loadmat("data/RN101/" + opt.dataset + "/" + opt.image_embedding + ".mat")
            label = matcontent['labels'].astype(int).squeeze() - 1
            if self.finetune:
                matcontent = sio.loadmat("data/RN101/" + opt.dataset + "/" + opt.image_embedding + "_finetuned.mat")
                # label = matcontent['labels'].astype(int).squeeze() - 1

            feature = matcontent['features'].T
            if opt.dataset == "APY" and self.finetune:
                feature = feature.T
        else:
            BACKBONEcontent = pickle.load( open( "data/{}_{}_data.p".format(opt.feature_backbone, opt.dataset.lower()), "rb" ) )
            print ('{} FEATURES'.format(opt.feature_backbone))
            feature = BACKBONEcontent['features']
            label = BACKBONEcontent['labels'].astype(int).squeeze() - 1

        matcontent = sio.loadmat("data/RN101/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # if opt.dataset == "FLO":
        #     temp_norm = torch.norm(self.attribute, p=2, dim=1).unsqueeze(1).expand_as(self.attribute)
        #     self.attribute = self.attribute.div(temp_norm + 1e-5)

        #
        #
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        #
        # train_att = matcontent['att_train']
        # seen_pro = matcontent['seen_pro']
        # attribute = matcontent['attribute']
        # unseen_pro = matcontent['unseen_pro']
        # self.attribute = torch.from_numpy(attribute).float()
        # self.train_att = seen_pro.astype(np.float32)
        # self.test_att = unseen_pro.astype(np.float32)

        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label