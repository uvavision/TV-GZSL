from builtins import breakpoint
import numpy as np
import scipy.misc
import urllib.request as urllib

import utils.dataloaders as dataloaders
from utils.helpers import *
import methods.inn_train as inn_zero_train

import torch
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable

# =======================================================
from functionalities import tracker as tk
from functionalities import MMD_autoencoder_loss as cl
# =======================================================
from models import encoder_template as et
from models import decoder_template as dt
from models.sdgzsl_model import *

from methods.SDGZSL import config as SDGZSL_CONFIG
from methods.SDGZSL import train as SDGZSL

from methods.FREE import config as FREE_CONFIG
from methods.FREE import train as FREE

import os

import pickle

# this linear classifier is just the same as CADA
class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Wrapper:
    """
    All steps for our Curriculum Learning approach can be called from here.

    Args:
        args (dictionary): all user defined parameters
    """

    def __init__(self, args):
        """
        Initiazile the Model with all the parameters predifined by the user - check for the command_line_example.py file for all variables -
        All possible configurations should be explicitly defined and passed through a dictionary (args) 

        Args:
            args (dictionary): all user defined parameters
        """
        self.args = args
        if self.args.feature_backbone in ['resnet101', 'resnet152', 'resnet50', 'resnet50_moco', 'adv_inception_v3', 
                                          'inception_v3', 'virtex', 'virtex2', 'dino_resnet50']:
            image_feature_size = 2048
        elif self.args.feature_backbone in ['googlenet', 'shufflenet', 'vit_large', 'resnet50_clip', 'resnet50x64_clip', 
                                            'mlp_mixer_l16', 'vit_large_21k']:
            image_feature_size = 1024
        elif self.args.feature_backbone in ['vgg16', 'alexnet']:
            image_feature_size = 4096
        elif self.args.feature_backbone in ['CLIP', 'resnet101_clip', 'vit_b32_clip', 'vit_b16_clip', 'vq_vae_fromScratch']:
            image_feature_size = 512
        elif self.args.feature_backbone in ['vit', 'resnet50x16_clip', 'vit_l14_clip', 'mlp_mixer', 'vit_base_21k', 'deit_base', 
                                            'dino_vitb16', 'soho', 
                                            'vit_l14_clip_finetune_v2', 
                                            'vit_l14_clip_finetune_classAndAtt', 
                                            'vit_l14_clip_finetune_class200Epochs', 
                                            'vit_l14_clip_finetune_trainsetAndgenerated_100Epochs', 
                                            'vit_l14_clip_finetune_trainsetAndgenerated_200Epochs',
                                            'vit_l14_clip_finetuned_classAndAtt_200Epochs', 
                                            'vit_l14_clip_finetuned_setAndgenerated_classAndAtt_100Epochs', 
                                            'vit_l14_clip_finetuned_setAndgenerated_classAndAtt_200Epochs',
                                            'clip_l14_finetune_classes_200epochs', 'clip_l14_finetun_atts_200epochs',
                                            'clip_l14_finetun_atts_200epochs',
                                            'clip_l14_finetune_classes_200epochs_frozenAllExc1Layer', 
                                            'clip_l14_finetun_atts_200epochs_frozenAllExc1Layer', 
                                            'clip_l14_finetune_classAndAtt_200epochs_frozenAllExc1Layer',
                                            'clip_l14_finetune_classes_200epochs_frozenTextE',
                                            'clip_l14_finetun_atts_200epochs_frozenTextE',
                                            'clip_l14_finetune_classAndAtt_200epochs_frozenTextE',
                                            'clip_l14_finetun_atts_fromMAT_200epochs',
                                            'clip_l14_finetun_classAndatts_fromMAT_200epochs',
                                            'clip_l14_finetun_class_fromMAT_200epochs']:
            image_feature_size = 768
        elif self.args.feature_backbone in ['resnet50x4_clip']:
            image_feature_size = 640
        elif self.args.feature_backbone in ['vit_huge', 'vit_large_finetune_classes_200epochs']:
            image_feature_size = 1280
        elif self.args.feature_backbone in ['biggan_138k_128size', 'biggan_100k_224size']:
            image_feature_size = 1536
            
        if self.args.dataset == 'CUB':
            text_feature_size = 312
        elif self.args.dataset == 'SUN':
            text_feature_size = 102
        elif self.args.dataset == 'AWA2':
            text_feature_size = 85

        if self.args.methods == 'UPPER_BOUND':
            print ("Train using seen features only and get the best accuracy")

        elif self.args.methods == 'CADA':
            self.I_model = None
            self.T_model = None

            self.I_encoder = et(image_feature_size, 64, (1560, 1660))
            self.I_decoder = dt(64, image_feature_size, (1560, 1660))
            self.T_encoder = et(text_feature_size, 64, (1450, 665))
            self.T_decoder = dt(64, text_feature_size, (1450, 665))

            # optimizer for all models
            self.model_optimizer = None

        elif self.args.methods == 'SDGZSL':
            # breakpoint()
            opt = SDGZSL_CONFIG.CONFIG(self.args.dataset, self.args.finetuned_features, self.args.feature_backbone, self.args.gpu)
            sdgzsl = SDGZSL.SDGZSL(opt)
            sdgzsl.train()
            exit()

        elif self.args.methods == 'FREE':
            # breakpoint()
            opt = FREE_CONFIG.CONFIG(self.args.dataset, self.args.finetuned_features, self.args.feature_backbone, self.args.gpu)
            FREE.FREE(opt)
            exit()
            
    def set_model_hyperparameters(self, latent_dim=1024):
        """
        Set model hyperparameters based on the user parameter selection

        1) Check CUDA availability
        2) Allow use of multiple GPUs

        Args:
            ema (bool, optional): if the model is a Teacher model or not. Defaults to False.
        """
        if self.args.doParallel:
            if self.args.methods == 'CADA':
                self.I_encoder = torch.nn.DataParallel(self.I_encoder)
                self.I_decoder = torch.nn.DataParallel(self.I_decoder)
                self.T_encoder = torch.nn.DataParallel(self.T_encoder)
                self.T_decoder = torch.nn.DataParallel(self.T_decoder)
            elif self.args.methods == 'SDGZSL':
                self.model = torch.nn.DataParallel(self.model)
                self.relationNet = torch.nn.DataParallel(self.relationNet)
                self.discriminator = torch.nn.DataParallel(self.discriminator)
                self.ae = torch.nn.DataParallel(self.ae)

        if torch.cuda.is_available():
            if self.args.methods == 'CADA':
                self.I_encoder = self.I_encoder.cuda()
                self.I_decoder = self.I_decoder.cuda()
                self.T_encoder = self.T_encoder.cuda()
                self.T_decoder = self.T_decoder.cuda()
            elif self.args.methods == 'SDGZSL':
                self.model = self.model.cuda()
                self.relationNet = self.relationNet.cuda()
                self.discriminator = self.discriminator.cuda()
                self.ae = self.ae.cuda()

            self.args.use_cuda = True
            self.args.device = 'cuda'
            # torch.backends.cudnn.benchmark = True # I personally prefer this one, but lets set deterministic True for the sake of reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.args.use_cuda = False
            self.args.device = 'cpu'

        self.args.lr = 0.00015 #1e-3
        self.args.l2_reg = 1e-6

        # MMD_autoencoder_loss parameters
        a_distr = 1 # 0
        a_rec = 1
        a_spar = 1
        a_disen = 1 # 0
        a_disc = False
        # latent_dim = 512
        loss_type = 'l1'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conditional = False
        disc_lst = None
        cont_min = None
        cont_max = None
        num_iter = None
        self.both_model_loss = cl.MMD_autoencoder_loss(a_distr=a_distr, a_rec=a_rec, 
                                                        a_spar=a_spar, a_disen=a_disen, 
                                                        a_disc=a_disc, latent_dim=latent_dim, 
                                                        loss_type=loss_type, device=device, 
                                                        conditional=conditional, disc_lst=disc_lst, 
                                                        cont_min=cont_min, cont_max=cont_max, 
                                                        num_iter=num_iter)

    def set_model_optimizer(self):
        """
        Set model optimizer based on user parameter selection

        1) Set SGD or Adam optimizer
        2) Set SWA if set (check you have downloaded the library using: pip install torchcontrib)
        3) Print if: Use ZCA preprocessing (sometimes useful for CIFAR10) or debug mode is on or off 
           (to check the model on the test set without taking decisions based on it -- all decisions are taken based on the validation set)
        """
        if self.args.methods == 'CADA':
            prRed ('... Adam optimizer ...')
            trainable_parameters = [p for p in self.I_encoder.parameters() if p.requires_grad]
            trainable_parameters += [p for p in self.I_decoder.parameters() if p.requires_grad]
            trainable_parameters += [p for p in self.T_encoder.parameters() if p.requires_grad]
            trainable_parameters += [p for p in self.T_decoder.parameters() if p.requires_grad]

            optimizer = torch.optim.Adam(trainable_parameters, lr=self.args.lr, betas=(0.8, 0.8),
                                        eps=1e-4, weight_decay=self.args.l2_reg)
        
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.1)

            self.model_optimizer = optimizer
            self.model_scheduler = scheduler

    def train(self):
        """
        Executes the train standard algorithm.

        """
        # breakpoint()
        if self.args.methods == 'UPPER_BOUND':
            self.cl = inn_zero_train.Zero_Shot_Train(self.args, None, None)
            return
        if self.args.methods == 'CADA':
            models = {"I_encoder": self.I_encoder, "I_decoder": self.I_decoder, "T_encoder": self.T_encoder, "T_decoder": self.T_decoder}
            optimizer = [self.model_optimizer]
        elif self.args.methods == 'SDGZSL':
            models = {"model": self.model, "relationNet": self.relationNet, "discriminator": self.discriminator, "ae": self.ae}
            optimizer = {"optimizer": self.optimizer, "relation_optimizer": self.relation_optimizer, "dis_optimizer": self.dis_optimizer, "ae_optimizer": self.ae_optimizer}
        self.cl = inn_zero_train.Zero_Shot_Train(self.args, models, optimizer)
        # train using only labeled subset
        self.cl.train_model()

    def train_classifier(self):

        img_seen_feat, img_seen_label = self.sample_train_data_on_sample_per_class_basis(
            self.cl.dataset_loaded.data['train_seen']['resnet_features'], 
            self.cl.dataset_loaded.data['train_seen']['labels'],
            250) # 125
        if self.args.methods == 'UPPER_BOUND':
            z_seen_img = self.cl.extract_raw_image_features(img_seen_feat)
        else:
            z_seen_img = self.cl.extract_z_image_features(img_seen_feat)

            att_seen_feat, att_seen_label = self.sample_train_data_on_sample_per_class_basis(
                self.cl.dataset_loaded.seenclass_aux_data,
                self.cl.dataset_loaded.seenclasses.long(),
                125)
            z_seen_att = self.cl.extract_z_aux_features(att_seen_feat)

            att_unseen_feat, att_unseen_label = self.sample_train_data_on_sample_per_class_basis(
                        self.cl.dataset_loaded.novelclass_aux_data,
                        self.cl.dataset_loaded.novelclasses.long(),
                        500)
            z_unseen_att = self.cl.extract_z_aux_features(att_unseen_feat)

            z_seen_img = torch.tensor(z_seen_img)
            z_unseen_att = torch.tensor(z_unseen_att)
            z_seen_att = torch.tensor(z_seen_att)

        if self.args.methods == 'UPPER_BOUND':
            val_features_imgs, val_targets_imgs = self.cl.extract_raw_features('test_seen', I_model=True)
        else:
            val_features_imgs, val_targets_imgs = self.cl.extract_z_features('test_seen', I_model=True)
        X_val = torch.tensor(val_features_imgs)
        Y_val = torch.tensor(val_targets_imgs)

        if self.args.methods == 'UPPER_BOUND':
            test_features_imgs, test_targets_imgs = self.cl.extract_raw_features('test_unseen', I_model=True)
        else:
            test_features_imgs, test_targets_imgs = self.cl.extract_z_features('test_unseen', I_model=True)
        X_test = torch.tensor(test_features_imgs)
        Y_test = torch.tensor(test_targets_imgs)

        if self.args.methods == 'UPPER_BOUND':
            X_train = z_seen_img
            Y_train = img_seen_label.cpu()
        else:
            X_train = torch.cat((z_seen_img, z_seen_att, z_unseen_att), dim=0)
            Y_train = torch.cat((img_seen_label.cpu(), att_seen_label.cpu(), att_unseen_label.cpu()), dim=0)


        print (' -- Adam Model -- ')
        adam_model = self.train_final_classifier(self.cl, X_train, Y_train, X_val, Y_val, X_test, Y_test, SGD=False)


    def train_final_classifier(self, cl, X_train, Y_train, X_val, Y_val, X_test, Y_test, SGD=True):
        num_classes = len(cl.dataset_loaded.seenclasses) + len(cl.dataset_loaded.novelclasses)
        train_loader, valid_loader, test_loader = dataloaders.get_training_classifier_dataloaders(self.args.workers, 100, X_train, Y_train, X_val, Y_val, X_test, Y_test)

        classifier_model = LINEAR_LOGSOFTMAX(X_train[0].shape[-1], num_classes)
        classifier_model = classifier_model.cuda()
        if SGD:
            classifier_model_optimizer = torch.optim.SGD(classifier_model.parameters(), 0.001,
                                                            momentum=self.args.momentum,
                                                            weight_decay=self.args.weight_decay,
                                                            nesterov=self.args.nesterov)
        else:
            classifier_model_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001, betas=(0.5, 0.999), amsgrad=True)

        criterion = classifier_model.lossfunction

        seen = cl.dataset_loaded.seenclasses
        unseen = cl.dataset_loaded.novelclasses

        for epochs in range (101):
            classifier_model.train()
            for b_idx, (data, target) in enumerate(train_loader):
                classifier_model_optimizer.zero_grad()

                data = data.cuda()
                target = target.cuda()
                output = classifier_model(data)
                loss = criterion(output, target)

                classifier_model_optimizer.zero_grad()
                loss.backward()
                classifier_model_optimizer.step()


            acc_seen, detailed_res_seen_test = self.validation_classifier(classifier_model, valid_loader, seen)
            acc_novel, detailed_res_unseen_test = self.validation_classifier(classifier_model, test_loader, unseen)
            acc_seen = acc_seen.data.item()
            acc_novel = acc_novel.data.item()
            H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
            if self.args.methods == 'UPPER_BOUND':
                print (epochs, 'Seen',  acc_seen)
            else:
                print (epochs, 'Seen',  acc_seen, 'Unseen',  acc_novel, 'Harmonic', H)
        
        return classifier_model

    def validation_classifier(self, classifier_model, loader, target_classes):
        classifier_model.eval()
        sm = nn.Softmax(dim=1)

        res = 0
        all_targets = []
        all_predictions = []
        all_values = []

        for b_idx, (data, target) in enumerate(loader):

            data = data.cuda()
            target = target.cuda()
            output = classifier_model(data)
            res += self.accuracy(output, target)[0].data.item()

            all_targets.extend(target.detach().cpu().numpy())
            all_predictions.extend(torch.argmax(output.data, 1).detach().cpu().numpy())
            all_values.extend(sm(output.data).detach().cpu().numpy())

        acc = self.compute_per_class_acc_gzsl(all_targets, all_predictions, target_classes)
        return acc, [all_targets, all_values]

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        
        per_class_accuracies = Variable(torch.zeros(target_classes.shape[0]).float().to(self.args.device)).detach()
        predicted_label = np.array(predicted_label)
        predicted_label = torch.tensor(predicted_label).to(self.args.device)
        test_label = np.array(test_label)
        test_label = torch.tensor(np.array(test_label)).to(self.args.device)

        for i in range(target_classes.shape[0]):
            is_class = test_label==target_classes[i]
            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum(),float(is_class.sum())) #torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())

        return per_class_accuracies.mean()

    def load_from_checkpoint(self, epoch):
        exp_dir = os.path.join(self.args.root_dir, '{}/{}'.format(self.args.dataset, self.args.add_name))
        print("=> loading pretrained checkpoint_I '{}'".format(exp_dir + '/{}.checkpoint_I.ckpt'.format(epoch)))
        print("=> loading pretrained checkpoint_T '{}'".format(exp_dir + '/{}.checkpoint_T.ckpt'.format(epoch)))
        #######
        print("=> loading pretrained checkpoint_IE '{}'".format(exp_dir + '/{}.checkpoint_IE.ckpt'.format(epoch)))
        print("=> loading pretrained checkpoint_ID '{}'".format(exp_dir + '/{}.checkpoint_ID.ckpt'.format(epoch)))
        print("=> loading pretrained checkpoint_TE '{}'".format(exp_dir + '/{}.checkpoint_TE.ckpt'.format(epoch)))
        print("=> loading pretrained checkpoint_TD '{}'".format(exp_dir + '/{}.checkpoint_TD.ckpt'.format(epoch)))
        #######
        checkpoint_IE = torch.load(exp_dir  + '/{}.checkpoint_IE.ckpt'.format(epoch))
        checkpoint_ID = torch.load(exp_dir  + '/{}.checkpoint_ID.ckpt'.format(epoch))
        checkpoint_TE = torch.load(exp_dir  + '/{}.checkpoint_TE.ckpt'.format(epoch))
        checkpoint_TD = torch.load(exp_dir  + '/{}.checkpoint_TD.ckpt'.format(epoch))
        #######
        self.I_encoder.load_state_dict(checkpoint_IE['state_dict'])
        self.I_decoder.load_state_dict(checkpoint_ID['state_dict'])
        self.T_encoder.load_state_dict(checkpoint_TE['state_dict'])
        self.T_decoder.load_state_dict(checkpoint_TD['state_dict'])   
        #######
        print("=> loaded pretrained checkpoint_IE '{}' (epoch {})".format(exp_dir + '/{}.checkpoint_IE.ckpt'.format(epoch), checkpoint_IE['epoch']))
        print("=> loaded pretrained checkpoint_ID '{}' (epoch {})".format(exp_dir + '/{}.checkpoint_ID.ckpt'.format(epoch), checkpoint_ID['epoch']))      
        print("=> loaded pretrained checkpoint_TE '{}' (epoch {})".format(exp_dir + '/{}.checkpoint_TE.ckpt'.format(epoch), checkpoint_TE['epoch']))
        print("=> loaded pretrained checkpoint_TD '{}' (epoch {})".format(exp_dir + '/{}.checkpoint_TD.ckpt'.format(epoch), checkpoint_TD['epoch']))      

    def sample_train_data_on_sample_per_class_basis(self, features, label, sample_per_class):
        sample_per_class = int(sample_per_class)
        if sample_per_class != 0 and len(label) != 0:
            classes = label.unique()
            for i, s in enumerate(classes):

                features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                # if number of selected features is smaller than the number of features we want per class:
                multiplier = torch.ceil(torch.cuda.FloatTensor([max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                if i == 0:
                    features_to_return = features_of_that_class[:sample_per_class, :]
                    labels_to_return = s.repeat(sample_per_class)
                else:
                    features_to_return = torch.cat(
                        (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                    labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)), dim=0)

            return features_to_return, labels_to_return
        else:
            return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res