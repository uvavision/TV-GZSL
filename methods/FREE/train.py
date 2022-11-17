#######################
#author: Shiming Chen
#FREE
#######################
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
#import functions
from .model import *
from .util import *
from .classifier import * 

# from config import opt
import time
# import classifier_cls as classifier2
from .center_loss import TripCenterLoss_min_margin,TripCenterLoss_margin

class FREE():
    def __init__(self, opt):
        if opt.manualSeed is None:
            opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", opt.manualSeed)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        if opt.cuda:
            torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True
        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        # load data
        data = DATA_LOADER(opt)
        print("# of training samples: ", data.ntrain)

        cls_criterion = nn.NLLLoss()
        if opt.dataset in ['CUB','FLO']:
            center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
        elif opt.dataset in ['AWA1','AWA2', 'APY']:
            center_criterion = TripCenterLoss_min_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
        elif opt.dataset in ['SUN']:
            center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)
        else:
            raise ValueError('Dataset %s is not supported'%(opt.dataset))

        netE = Encoder(opt)
        netG = Generator(opt)
        netD = Discriminator(opt)
        netFR = FR(opt, opt.attSize)

        print(netE)
        print(netG)
        print(netD)
        print(netFR)
        
        ###########
        # Init Tensors
        input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
        input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
        noise = torch.FloatTensor(opt.batch_size, opt.nz)
        input_label = torch.LongTensor(opt.batch_size)
        #one = torch.FloatTensor([1])
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        # beta=0
        ##########
        # Cuda
        if opt.cuda:
            netD.cuda()
            netE.cuda()
            netG.cuda()

            netFR.cuda()
            self.input_res = input_res.cuda()
            self.noise, self.input_att = noise.cuda(), input_att.cuda()
            self.one = one.cuda()
            self.mone = mone.cuda()
            self.input_label=input_label.cuda()
    
        self.optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
        self.optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
        self.optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
        self.optimizerFR      = optim.Adam(netFR.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
        self.optimizer_center   = optim.Adam(center_criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

        self.netE = netE
        self.netD = netD
        self.netG = netG
        self.netFR = netFR
        self.center_criterion = center_criterion
        self.data = data
        self.opt = opt

        self.train_free(opt)

    def train_free(self, opt):
        if not os.path.exists(os.path.join(opt.result_root, opt.dataset)):
            os.makedirs(os.path.join(opt.result_root, opt.dataset))

        best_gzsl_acc = 0
        best_zsl_acc = 0
        for epoch in range(0,opt.nepoch):
            for loop in range(0,opt.loop):
                mean_lossD = 0
                mean_lossG = 0
                for i in range(0, self.data.ntrain, opt.batch_size):
                    #########Discriminator training ##############
                    for p in self.netD.parameters(): #unfreeze discrimator
                        p.requires_grad = True

                    for p in self.netFR.parameters(): #unfreeze deocder
                        p.requires_grad = True
                    # Train D1 and Decoder (and Decoder Discriminator)
                    gp_sum = 0 #lAMBDA VARIABLE
                    for iter_d in range(opt.critic_iter):
                        self.sample()
                        self.netD.zero_grad()          
                        input_resv = Variable(self.input_res)
                        input_attv = Variable(self.input_att)
                        
                        if opt.encoded_noise:        
                            means, log_var = self.netE(input_resv, input_attv)
                            std = torch.exp(0.5 * log_var)
                            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                            eps = Variable(eps.cuda())
                            z = eps * std + means #torch.Size([64, 312])
                        else:
                            self.noise.normal_(0, 1)
                            z = Variable(self.noise)
                        
                        ################# update FR
                        self.netFR.zero_grad()
                        muR, varR, criticD_real_FR, latent_pred, _, recons_real = self.netFR(input_resv)
                        criticD_real_FR = criticD_real_FR.mean()
                        R_cost = opt.recons_weight*self.WeightedL1(recons_real, input_attv) 
                        
                        fake = self.netG(z, c=input_attv)
                        muF, varF, criticD_fake_FR, _, _, recons_fake= self.netFR(fake.detach())
                        criticD_fake_FR = criticD_fake_FR.mean()
                        gradient_penalty = self.calc_gradient_penalty_FR(self.netFR, input_resv, fake.data)
                        center_loss_real=self.center_criterion(muR, self.input_label,margin=opt.center_margin, incenter_weight=opt.incenter_weight)
                        D_cost_FR = center_loss_real*opt.center_weight + R_cost
                        D_cost_FR.backward()
                        self.optimizerFR.step()
                        self.optimizer_center.step()
                        
                        ############################
                        
                        criticD_real = self.netD(input_resv, input_attv)
                        criticD_real = opt.gammaD*criticD_real.mean()
                        criticD_real.backward(self.mone)
                        
                        criticD_fake = self.netD(fake.detach(), input_attv)
                        criticD_fake = opt.gammaD*criticD_fake.mean()
                        criticD_fake.backward(self.one)
                        # gradient penalty
                        gradient_penalty = opt.gammaD*self.calc_gradient_penalty(self.netD, self.input_res, fake.data, self.input_att)

                        gp_sum += gradient_penalty.data
                        gradient_penalty.backward()         
                        Wasserstein_D = criticD_real - criticD_fake
                        D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae 
                        self.optimizerD.step()

                    gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
                    if (gp_sum > 1.05).sum() > 0:
                        opt.lambda1 *= 1.1
                    elif (gp_sum < 1.001).sum() > 0:
                        opt.lambda1 /= 1.1

                    #############Generator training ##############
                    # Train Generator and Decoder
                    for p in self.netD.parameters(): #freeze discrimator
                        p.requires_grad = False
                    if opt.recons_weight > 0 and opt.freeze_dec:
                        for p in self.netFR.parameters(): #freeze decoder
                            p.requires_grad = False

                    self.netE.zero_grad()
                    self.netG.zero_grad()

                    input_resv = Variable(self.input_res)
                    input_attv = Variable(self.input_att)
                    means, log_var = self.netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means #torch.Size([64, 312])

                    recon_x = self.netG(z, c=input_attv)
                    vae_loss_seen = self.loss_fn(recon_x, input_resv, means, log_var) 
                    errG = vae_loss_seen
                    
                    if opt.encoded_noise:
                        criticG_fake = self.netD(recon_x,input_attv).mean()
                        fake = recon_x 
                    else:
                        self.noise.normal_(0, 1)
                        noisev = Variable(self.noise)

                        fake = self.netG(noisev, c=input_attv)
                        criticG_fake = self.netD(fake,input_attv).mean()
                        

                    G_cost = -criticG_fake
                    errG += opt.gammaG*G_cost
                    
                    ######################################original
                    self.netFR.zero_grad()
                    _,_,criticG_fake_FR,latent_pred_fake, _, recons_fake = self.netFR(fake, train_G=True)
                    R_cost = self.WeightedL1(recons_fake, input_attv)
                    errG += opt.recons_weight * R_cost
                    

                    
                    errG.backward()
                    # write a condition here
                    self.optimizer.step()
                    self.optimizerG.step()

                    self.optimizerFR.step() 
                
            print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),vae_loss_seen.item()))#,end=" ")
            
            if epoch % 100 == 0:
                self.netG.eval()
                self.netFR.eval()

                syn_feature, syn_label = self.generate_syn_feature(self.netG,self.data.unseenclasses, self.data.attribute, opt.syn_num)

                ### Concatenate real seen features with synthesized unseen features
                train_X = torch.cat((self.data.train_feature, syn_feature), 0)
                train_Y = torch.cat((self.data.train_label, syn_label), 0)
                nclass = opt.nclass_all
                ### Train GZSL classifier
                gzsl_cls = CLASSIFIER(train_X, train_Y, self.data, nclass, opt.cuda, opt.classifier_lr, 0.5,25, opt.syn_num, netFR=self.netFR, dec_size=opt.attSize, dec_hidden_size=(opt.latensize*2))
                
                if best_gzsl_acc <= gzsl_cls.H:
                    best_gzsl_epoch= epoch
                    best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
                    ### torch.save({'self.netG': self.netG.state_dict()}, os.path.join(opt.result_root, opt.dataset, 'checkpoint_G.pth.tar'))
                    ### torch.save({'self.netFR': self.netFR.state_dict()}, os.path.join(opt.result_root, opt.dataset, 'checkpoint_F.pth.tar'))
                print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")
                    
                if epoch % 10 == 0:
                    print('GZSL: epoch=%d, best_seen=%.3f, best_unseen=%.3f, best_h=%.3f' % (best_gzsl_epoch, best_acc_seen, best_acc_unseen, best_gzsl_acc))
                # print('ZSL: epoch=%d, best unseen accuracy=%.4f' % (best_zsl_epoch, best_zsl_acc))
                
                
                # reset G to training mode
                self.netG.train()
                self.netFR.train()

        # print('feature(X+feat1): 2048+4096')
        print('softmax: feature(X+feat1+feat2): 8494')
        print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('Dataset', opt.dataset)
        # print('the best ZSL unseen accuracy is', best_zsl_acc)
        # if opt.gzsl:
        print('Dataset', opt.dataset)
        print('the best GZSL seen accuracy is', best_acc_seen)
        print('the best GZSL unseen accuracy is', best_acc_unseen)
        print('the best GZSL H is', best_gzsl_acc)

    def loss_fn(self, recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
        BCE = BCE.sum()/ x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
        #return (KLD)
        return (BCE + KLD)
            
    def sample(self):
        batch_feature, batch_label, batch_att = self.data.next_seen_batch(self.opt.batch_size)
        self.input_res.copy_(batch_feature)
        self.input_att.copy_(batch_att)
        self.input_label.copy_(map_label(batch_label, self.data.seenclasses))

    def WeightedL1(self, pred, gt):
        wt = (pred-gt).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
        loss = wt * (pred-gt).abs()
        return loss.sum()/loss.size(0)
        
    def generate_syn_feature(self, generator,classes, attribute,num):
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass*num, self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num) 
        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            with torch.no_grad():
                syn_noisev = Variable(syn_noise)
                syn_attv = Variable(syn_att)
            fake = generator(syn_noisev,c=syn_attv)
            output = fake
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

        return syn_feature, syn_label

    def calc_gradient_penalty(self, netD,real_data, fake_data, input_att):
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.opt.cuda:
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates, Variable(input_att))
        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    def calc_gradient_penalty_FR(self, netFR, real_data, fake_data):
        #print real_data.size()
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)
        _,_,disc_interpolates,_ ,_, _ = netFR(interpolates)
        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty


    def MI_loss(self, mus, sigmas, i_c, alpha=1e-8):
        kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                    - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

        MI_loss = (torch.mean(kl_divergence) - i_c)

        return MI_loss

    def optimize_beta(self, beta, MI_loss,alpha2=1e-6):
        beta_new = max(0, beta + (alpha2 * MI_loss))

        # return the updated beta value:
        return beta_new



