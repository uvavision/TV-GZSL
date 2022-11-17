import os
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from utils.logger import Logger
from utils.helpers import *
from utils.scheduler_ramps import *
from .base import *
from utils.cada_dataloader import *

# from https://github.com/pytorch/contrib = pip install torchcontrib
from torchcontrib.optim import SWA

from sklearn.metrics.pairwise import euclidean_distances
from functionalities import loss as lo

class Zero_Shot_Train(Train_Base):
    def __init__(self, args, models, optimizers):
        """
        """
        self.best_prec1 = 0

        ### list error and losses ###
        self.train_class_loss_list = []
        self.train_error_list = []
        self.train_lr_list = []
        self.val_class_loss_list = []
        self.val_error_list = []

        exp_dir = os.path.join(args.root_dir, '{}/{}'.format(args.dataset, args.add_name))
        prGreen('Results will be saved to this folder: {}'.format(exp_dir))

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        self.args = args
        self.args.exp_dir = exp_dir


        if self.args.methods == 'CADA':
            self.I_encoder = models['I_encoder'] 
            self.I_decoder = models['I_decoder'] 
            self.T_encoder = models['T_encoder'] 
            self.T_decoder = models['T_decoder']
            self.model_optimizer = optimizers[0]
        elif self.args.methods == 'SDGZSL':
            self.model = models['model']
            self.relationNet = models['relationNet']
            self.discriminator = models['discriminator']
            self.ae = models['ae']
            self.optimizer = optimizers['optimizer']
            self.relation_optimizer = optimizers['relation_optimizer']
            self.dis_optimizer = optimizers['dis_optimizer']
            self.ae_optimizer = optimizers['ae_optimizer']
            self.ones = torch.ones(self.args.batch_size, dtype=torch.long, device=self.args.device)
            self.zeros = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.args.device)
            self.mse = nn.MSELoss().to(self.args.device)

        self.dataset_loaded = DATA_LOADER(self.args.feature_backbone, self.args.finetuned_features, dataset=self.args.dataset)

        # CADA HYPERPARAMETERS
        self.warmup = {'beta': {'factor': 0.25,
                                'end_epoch': 93,
                                'start_epoch': 0},
                        'cross_reconstruction': {'factor': 2.37,
                                                'end_epoch': 75,
                                                'start_epoch': 21},
                        'distance': {'factor': 8.13,
                                    'end_epoch': 22,
                                    'start_epoch': 6}}

        # self.reconstruction_criterion = nn.MSELoss(size_average=False)
        self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def train_model(self, init_epoch = 0): #, trainloader, validloader, testloader, model, optimizer, train_logger, val_logger, num_classes = 10, init_epoch = 0):
        """
        Method to train the chosen method.
        """
        dataset = DATA_LOADER(self.args.feature_backbone, self.args.finetuned_features, dataset=self.args.dataset)
        for epoch in range(init_epoch, self.args.epochs + (self.args.lr_rampdown_epochs-self.args.epochs)):
            start_time = time.time()

            # use scheduler to manage lr
            # self.model_scheduler.step()
            if self.args.methods == 'CADA':
                self.train_base(epoch, dataset)
            elif self.args.methods == 'SDGZSL':
                self.train_sdgzsl(epoch, dataset)

            print("--- training epoch in %s seconds ---" % (time.time() - start_time))

            ## not usefull for now
            # if epoch % self.args.checkpoint_epochs == 0:
            #     self.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': self.I_encoder.state_dict(),
            #     }, False, self.args.exp_dir, epoch + 1, 'IE')

            #     self.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': self.I_decoder.state_dict(),
            #     }, False, self.args.exp_dir, epoch + 1, 'ID')

            #     self.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': self.T_encoder.state_dict(),
            #     }, False, self.args.exp_dir, epoch + 1, 'TE')

            #     self.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': self.T_decoder.state_dict(),
            #     }, False, self.args.exp_dir, epoch + 1, 'TD')

    def train_base(self, epoch, dataset, latent_dim=2048):
        """
        Train.
        """
        meters = AverageMeterSet()

        self.I_encoder.train()
        self.I_decoder.train()
        self.T_encoder.train()
        self.T_decoder.train()
        end = time.time()

        for loader_index, iters in enumerate(range(0, dataset.ntrain, self.args.batch_size)):

            label, data_from_modalities = dataset.next_batch(self.args.batch_size)

            label = label.long().to(self.args.device)
            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(self.args.device)
                data_from_modalities[j].requires_grad = False

            # measure data loading time
            meters.update('data_time', time.time() - end)

            # update lr based on epoch
            if epoch <= self.args.epochs:
                lr = self.adjust_learning_rate(self.model_optimizer, epoch, loader_index, dataset.ntrain)
            meters.update('lr', self.model_optimizer.param_groups[0]['lr'])

            x_1 = data_from_modalities[0]
            x_2 = data_from_modalities[1]

            x_1 = x_1.cuda()
            x_2 = x_2.cuda()

            
            _target_categorical = []
            for y in label:
                _target_categorical.append(np.eye(len(dataset.allclasses), dtype='float')[(dataset.seenclasses == y.data.item()).nonzero().item()])
            y = torch.tensor(np.array(_target_categorical)).cuda()
            y_clean = y.clone()

            batch_size = len(data_from_modalities[0])

            self.model_optimizer.zero_grad()

            I_lat_img_copy = x_1.clone()
            T_lat_img_copy = x_2.clone()

            I_mu, I_logvar = self.I_encoder(I_lat_img_copy)
            I_z_from_modality = reparameterize(True, I_mu, I_logvar)
            I_lat_from_lat = self.I_decoder(I_z_from_modality)

            T_mu, T_logvar = self.T_encoder(T_lat_img_copy)
            T_z_from_modality = reparameterize(True, T_mu, T_logvar)
            T_lat_from_lat = self.T_decoder(T_z_from_modality)


            reconstruction_loss = self.reconstruction_criterion(I_lat_from_lat, I_lat_img_copy)
            reconstruction_loss += self.reconstruction_criterion(T_lat_from_lat, T_lat_img_copy)

            I_lat_from_lat_fromT = self.I_decoder(T_z_from_modality)
            T_lat_from_lat_fromI = self.T_decoder(I_z_from_modality)

            cross_reconstruction_loss = self.reconstruction_criterion(I_lat_from_lat_fromT, I_lat_img_copy)
            cross_reconstruction_loss += self.reconstruction_criterion(T_lat_from_lat_fromI, T_lat_img_copy)

            KLD = 0.5 * torch.sum(1 + I_logvar - I_mu.pow(2) - I_logvar.exp())
            KLD += 0.5 * torch.sum(1 + T_logvar - T_mu.pow(2) - T_logvar.exp())

            distance = torch.sum((I_mu - T_mu) ** 2, dim=1)
            distance += torch.sum((torch.sqrt(I_logvar.exp()) - torch.sqrt(T_logvar.exp())) ** 2, dim=1)
            distance = torch.sqrt(distance)
            distance = distance.sum()


            f1 = 1.0*(epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
            f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
            cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

            f2 = 1.0 * (epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
            f2 = f2 * (1.0 * self.warmup['beta']['factor'])
            beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

            f3 = 1.0*(epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
            f3 = f3*(1.0*self.warmup['distance']['factor'])
            distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

            self.model_optimizer.zero_grad()
            latent_loss = reconstruction_loss - beta * KLD
            if cross_reconstruction_loss>0:
                latent_loss += cross_reconstruction_factor*cross_reconstruction_loss
            if distance_factor >0:
                latent_loss += distance_factor*distance

            meters.update('latent_loss', latent_loss.data.item(), batch_size)
            
            latent_loss.backward()

            self.model_optimizer.step()

            meters.update('batch_time', time.time() - end)
            end = time.time()


        print (epoch, '- Latent Loss alone', meters['latent_loss'].avg ) 

    def train_sdgzsl(self, epoch, dataset, latent_dim=2048):
        """
        Train.
        """
        meters = AverageMeterSet()

        self.model.train()
        self.relationNet.train()
        self.discriminator.train()
        self.ae.train()
        end = time.time()

        for loader_index, iters in enumerate(range(0, dataset.ntrain, self.args.batch_size)):

            label, data_from_modalities = dataset.next_batch(self.args.batch_size)

            label = label.long().to(self.args.device)
            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(self.args.device)
                data_from_modalities[j].requires_grad = False

            # measure data loading time
            meters.update('data_time', time.time() - end)

            # update lr based on epoch
            if epoch <= self.args.epochs:
                lr = self.adjust_learning_rate(self.model_optimizer, epoch, loader_index, dataset.ntrain)
            meters.update('lr', self.model_optimizer.param_groups[0]['lr'])

            x_1 = data_from_modalities[0]
            x_2 = data_from_modalities[1]

            x_1 = x_1.cuda()
            x_2 = x_2.cuda()

            
            _target_categorical = []
            for y in label:
                _target_categorical.append(np.eye(len(dataset.allclasses), dtype='float')[(dataset.seenclasses == y.data.item()).nonzero().item()])
            y = torch.tensor(np.array(_target_categorical)).cuda()
            y_clean = y.clone()

            batch_size = len(data_from_modalities[0])

            self.model_optimizer.zero_grad()

            I_lat_img_copy = x_1.clone()
            T_lat_img_copy = x_2.clone()

            I_mu, I_logvar = self.I_encoder(I_lat_img_copy)
            I_z_from_modality = reparameterize(True, I_mu, I_logvar)
            I_lat_from_lat = self.I_decoder(I_z_from_modality)

            T_mu, T_logvar = self.T_encoder(T_lat_img_copy)
            T_z_from_modality = reparameterize(True, T_mu, T_logvar)
            T_lat_from_lat = self.T_decoder(T_z_from_modality)


            reconstruction_loss = self.reconstruction_criterion(I_lat_from_lat, I_lat_img_copy)
            reconstruction_loss += self.reconstruction_criterion(T_lat_from_lat, T_lat_img_copy)

            I_lat_from_lat_fromT = self.I_decoder(T_z_from_modality)
            T_lat_from_lat_fromI = self.T_decoder(I_z_from_modality)

            cross_reconstruction_loss = self.reconstruction_criterion(I_lat_from_lat_fromT, I_lat_img_copy)
            cross_reconstruction_loss += self.reconstruction_criterion(T_lat_from_lat_fromI, T_lat_img_copy)

            KLD = 0.5 * torch.sum(1 + I_logvar - I_mu.pow(2) - I_logvar.exp())
            KLD += 0.5 * torch.sum(1 + T_logvar - T_mu.pow(2) - T_logvar.exp())

            distance = torch.sum((I_mu - T_mu) ** 2, dim=1)
            distance += torch.sum((torch.sqrt(I_logvar.exp()) - torch.sqrt(T_logvar.exp())) ** 2, dim=1)
            distance = torch.sqrt(distance)
            distance = distance.sum()


            f1 = 1.0*(epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
            f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
            cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

            f2 = 1.0 * (epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
            f2 = f2 * (1.0 * self.warmup['beta']['factor'])
            beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

            f3 = 1.0*(epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
            f3 = f3*(1.0*self.warmup['distance']['factor'])
            distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

            self.model_optimizer.zero_grad()
            latent_loss = reconstruction_loss - beta * KLD
            if cross_reconstruction_loss>0:
                latent_loss += cross_reconstruction_factor*cross_reconstruction_loss
            if distance_factor >0:
                latent_loss += distance_factor*distance

            meters.update('latent_loss', latent_loss.data.item(), batch_size)
            
            latent_loss.backward()

            self.model_optimizer.step()

            meters.update('batch_time', time.time() - end)
            end = time.time()


        print (epoch, '- Latent Loss alone', meters['latent_loss'].avg )

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):

        per_class_accuracies = Variable(torch.zeros(target_classes.shape[0]).float().to(self.args.device)).detach()
        predicted_label = np.array(predicted_label)
        predicted_label = torch.tensor(predicted_label).to(self.args.device)

        test_label = torch.tensor(np.array(test_label)).to(self.args.device)

        for i in range(target_classes.shape[0]):
            is_class = test_label==target_classes[i]
            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum(),float(is_class.sum())) #torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())

        return per_class_accuracies.mean()

    def extract_z_image_features(self, features, latent_dim=2048):
        self.I_encoder.eval()

        converted_features = []

        total_samples = features.shape[0]
        for i in range (0, total_samples, 128):
            x = features[i:i+128]
            x = x.cuda()

            I_mu, I_logvar = self.I_encoder(x)
            I_z_from_modality = reparameterize(True, I_mu, I_logvar)
            converted_features.extend(I_z_from_modality.cpu().detach().numpy())

        return converted_features

    def extract_raw_image_features(self, features, latent_dim=2048):
        converted_features = []

        total_samples = features.shape[0]
        for i in range (0, total_samples, 128):
            x = features[i:i+128]
            converted_features.extend(x.cpu().detach().numpy())

        return converted_features

    def extract_z_aux_features(self, features, latent_dim=2048):
        self.T_encoder.eval()

        converted_features = []

        total_samples = features.shape[0]
        for i in range (0, total_samples, 128):
            x = features[i:i+128]
            x = x.cuda()

            T_mu, T_logvar = self.T_encoder(x)
            T_z_from_modality = reparameterize(True, T_mu, T_logvar)
            converted_features.extend(T_z_from_modality.cpu().detach().numpy())

        return converted_features

    def extract_z_features(self, loader, I_model=True, latent_dim=2048):
        self.I_encoder.eval()
        self.T_encoder.eval()

        if I_model:
            lookup_modality = 'resnet_features'
        else:
            self.T_model.eval()
            lookup_modality = 'attributes'

        features = []
        targets = []

        total_samples = self.dataset_loaded.data[loader][lookup_modality].shape[0]

        for i in range (0, total_samples, 128):
            x = self.dataset_loaded.data[loader][lookup_modality][i:i+128]
            if not I_model:
                x = x

            label = self.dataset_loaded.data[loader]['labels'][i:i+128]
            x = x.cuda()
            batch_size = len(x)
            total = batch_size


            if I_model:
                I_mu, I_logvar = self.I_encoder(x)
                I_z_from_modality = reparameterize(False, I_mu, I_logvar)
                z_2 = I_z_from_modality.cpu().detach().numpy()
            else:
                T_lat_img = self.T_model(x)
                T_lat_shape = T_lat_img.shape
                T_lat_img = T_lat_img.view(T_lat_img.size(0), -1)

                # add some noise
                zeros_noise_scale = 5e-2
                # attribute modality
                T_lat_img_mod = torch.cat([T_lat_img[:, :latent_dim], zeros_noise_scale * torch.randn((T_lat_img[:, latent_dim:]).shape).to(self.args.device)], dim=1)
                z_2 = T_lat_img[:, :latent_dim].cpu().detach().numpy()

            features.extend(z_2)
            targets.extend(label)

        return features, targets

    def extract_raw_features(self, loader, I_model=True, latent_dim=2048):

        if I_model:
            lookup_modality = 'resnet_features'
        else:
            lookup_modality = 'attributes'

        features = []
        targets = []

        # breakpoint()
        total_samples = self.dataset_loaded.data[loader][lookup_modality].shape[0]

        for i in range (0, total_samples, 128):
            x = self.dataset_loaded.data[loader][lookup_modality][i:i+128]
            if not I_model:
                x = x

            label = self.dataset_loaded.data[loader]['labels'][i:i+128]
            x = x.cuda()
            batch_size = len(x)
            total = batch_size

            features.extend(x.cpu().detach().numpy())
            targets.extend(label)

        return features, targets

    def get_features_for_classifier(self):
        cls_seenclasses = self.dataset_loaded.seenclasses
        cls_novelclasses = self.dataset_loaded.novelclasses


        train_seen_feat = self.dataset_loaded.data['train_seen']['resnet_features']
        train_seen_label = self.dataset_loaded.data['train_seen']['labels']

        novelclass_aux_data = self.dataset_loaded.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset_loaded.seenclass_aux_data

        novel_corresponding_labels = self.dataset_loaded.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset_loaded.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset_loaded.data['test_unseen'][
            'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset_loaded.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset_loaded.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset_loaded.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset_loaded.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset_loaded.data['train_unseen']['labels']


def reparameterize(reparameterize_with_noise, mu, logvar):
    if reparameterize_with_noise:
        sigma = torch.exp(logvar)
        eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
        eps  = eps.expand(sigma.size())
        return mu + sigma*eps
    else:
        return mu