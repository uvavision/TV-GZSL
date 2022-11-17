from builtins import breakpoint
import torch.optim as optim
import glob
import json
import argparse
import os
import random
import math
from time import gmtime, strftime
from .models import *
from .dataset import DATA_LOADER
from .utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.backends.cudnn as cudnn
from .classifier import CLASSIFIER

import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch


class SDGZSL():
    def __init__(self, opt):
        if opt.manualSeed is None:
            opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", opt.manualSeed)
        np.random.seed(opt.manualSeed)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed_all(opt.manualSeed)

        cudnn.benchmark = True
        print('Running parameters:')
        print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
        opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")
        
        self.opt = opt


    def train(self):
        # breakpoint()
        dataset = DATA_LOADER(self.opt)
        self.opt.C_dim = dataset.att_dim
        self.opt.X_dim = dataset.feature_dim
        self.opt.Z_dim = self.opt.latent_dim
        self.opt.y_dim = dataset.ntrain_class
        out_dir = 'SDGZSL_out/{}/wd-{}_b-{}_g-{}_lr-{}_sd-{}_dis-{}_nS-{}_nZ-{}_bs-{}_seed-{}_featBack-{}'.format(self.opt.dataset, self.opt.weight_decay,
                        self.opt.beta, self.opt.ga, self.opt.lr, self.opt.S_dim, self.opt.dis, self.opt.nSample, self.opt.Z_dim, self.opt.batchsize, self.opt.manualSeed, self.opt.feature_backbone)
        os.makedirs(out_dir, exist_ok=True)
        print("The output dictionary is {}".format(out_dir))

        log_dir = out_dir + '/log_{}.txt'.format(self.opt.dataset)
        with open(log_dir, 'w') as f:
            f.write('Training Start:')
            f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

        # breakpoint()
        dataset.feature_dim = dataset.train_feature.shape[1]
        self.opt.X_dim = dataset.feature_dim
        self.opt.Z_dim = self.opt.latent_dim
        self.opt.y_dim = dataset.ntrain_class

        data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), self.opt)

        self.opt.niter = int(dataset.ntrain/self.opt.batchsize) * self.opt.gen_nepoch
        self.opt.evl_interval = self.opt.niter

        result_gzsl_soft = Result()
        # result_zsl_soft = Result()

        model = VAE(self.opt).to(self.opt.gpu)
        relationNet = RelationNet(self.opt).to(self.opt.gpu)
        discriminator = Discriminator(self.opt).to(self.opt.gpu)
        ae = AE(self.opt).to(self.opt.gpu)
        print(model)

        with open(log_dir, 'a') as f:
            f.write('\n')
            f.write('Generative Model Training Start:')
            f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

        start_step = 0
        optimizer = optim.Adam(model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        relation_optimizer = optim.Adam(relationNet.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        ae_optimizer = optim.Adam(ae.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        ones = torch.ones(self.opt.batchsize, dtype=torch.long, device=self.opt.gpu)
        zeros = torch.zeros(self.opt.batchsize, dtype=torch.long, device=self.opt.gpu)
        mse = nn.MSELoss().to(self.opt.gpu)


        iters = math.ceil(dataset.ntrain/self.opt.batchsize)
        beta = 0.01
        coin = 0
        gamma = 0
        for it in range(start_step, self.opt.niter+1):

            if it % iters == 0:
                beta = min(self.opt.kl_warmup*(it/iters), 1)
                gamma = min(self.opt.tc_warmup * (it / iters), 1)

            blobs = data_layer.forward()
            feat_data = blobs['data']
            labels_numpy = blobs['labels'].astype(int)
            labels = torch.from_numpy(labels_numpy.astype('int')).to(self.opt.gpu)

            # breakpoint()
            C = np.array([dataset.train_att[i,:] for i in labels])
            C = torch.from_numpy(C.astype('float32')).to(self.opt.gpu)
            X = torch.from_numpy(feat_data).to(self.opt.gpu)
            sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(self.opt.gpu)
            sample_C_n = labels.unique().shape[0]
            sample_label = labels.unique().cpu()

            x_mean, z_mu, z_var, z = model(X, C)
            loss, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

            sample_labels = np.array(sample_label)
            re_batch_labels = []
            for label in labels_numpy:
                index = np.argwhere(sample_labels == label)
                re_batch_labels.append(index[0][0])
            re_batch_labels = torch.LongTensor(re_batch_labels)
            one_hot_labels = torch.zeros(self.opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(self.opt.gpu)

            # one_hot_labels = torch.tensor(
            #     torch.zeros(self.opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1)).to(self.opt.gpu)

            x1, h1, hs1, hn1 = ae(x_mean)
            relations = relationNet(hs1, sample_C)
            relations = relations.view(-1, labels.unique().cpu().shape[0])

            p_loss = self.opt.ga * mse(relations, one_hot_labels)

            x2, h2, hs2, hn2 = ae(X)
            relations = relationNet(hs2, sample_C)
            relations = relations.view(-1, labels.unique().cpu().shape[0])

            p_loss = p_loss + self.opt.ga * mse(relations, one_hot_labels)

            rec = mse(x1, X) + mse(x2, X)
            if coin > 0:
                s_score = discriminator(h1)
                tc_loss = self.opt.beta * gamma *((s_score[:, :1] - s_score[:, 1:]).mean())
                s_score = discriminator(h2)
                tc_loss = tc_loss + self.opt.beta * gamma* ((s_score[:, :1] - s_score[:, 1:]).mean())

                loss = loss + p_loss + rec + tc_loss
                coin -= 1
            else:
                s, n = permute_dims(hs1, hn1)
                b = torch.cat((s, n), 1).detach()
                s_score = discriminator(h1)
                n_score = discriminator(b)
                tc_loss = self.opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

                s, n = permute_dims(hs2, hn2)
                b = torch.cat((s, n), 1).detach()
                s_score = discriminator(h2)
                n_score = discriminator(b)
                tc_loss = tc_loss + self.opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

                dis_optimizer.zero_grad()
                tc_loss.backward(retain_graph=True)
                dis_optimizer.step()

                loss = loss + p_loss + rec
                coin += self.opt.dis_step
            optimizer.zero_grad()
            relation_optimizer.zero_grad()
            ae_optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            relation_optimizer.step()
            ae_optimizer.step()

            if it % self.opt.disp_interval == 0 and it:
                log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; p_loss:{:.3f}; rec:{:.3f}; tc:{:.3f}; gamma:{:.3f};'.format(it,
                                                self.opt.niter, loss.item(),kl.item(),p_loss.item(),rec.item(), tc_loss.item(), gamma)
                log_print(log_text, log_dir)

            if it % self.opt.evl_interval == 0 and it > self.opt.evl_start:
                model.eval()
                ae.eval()
                gen_feat, gen_label = synthesize_feature_test(model, ae, dataset, self.opt)
                with torch.no_grad():
                    train_feature = ae.encoder(dataset.train_feature.to(self.opt.gpu))[:,:self.opt.S_dim].cpu()
                    test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(self.opt.gpu))[:,:self.opt.S_dim].cpu()
                    test_seen_feature = ae.encoder(dataset.test_seen_feature.to(self.opt.gpu))[:,:self.opt.S_dim].cpu()

                train_X = torch.cat((train_feature, gen_feat), 0)
                train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
                """ GZSL"""
                cls = CLASSIFIER(self.opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, self.opt.classifier_lr, 0.5,
                                            self.opt.classifier_steps, self.opt.nSample, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print("Seen {:.2f}%  Novel {:.2f}%  Harmonic {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    cls.acc_seen, cls.acc_unseen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

                if result_gzsl_soft.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, model, self.opt.manualSeed, log_text,
                            out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                                result_gzsl_soft.best_acc_S_T,
                                                                                                result_gzsl_soft.best_acc_U_T))
                model.train()
                ae.train()
            if it % self.opt.save_interval == 0 and it:
                save_model(it, model, self.opt.manualSeed, log_text,
                        out_dir + '/Iter_{:d}.tar'.format(it))
                print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs

