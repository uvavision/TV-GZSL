import os, errno
import numpy as np
import random
import pickle

all_methods = ['DEVISE', 'ESZSL', 'ALE',		
                'CADA', 'tfVAEGAN', 'CE',
                'SDGZSL', 'FREE', 'UPPER_BOUND']

all_possible_archs_and_ft_features = ['resnet101', 'resnet152', 'resnet50', 'resnet50_moco', 'googlenet', 'vgg16', 'alexnet', 
                                    'shufflenet', 'vit', 'vit_large', 'adv_inception_v3', 'inception_v3', 
                                    'resnet50_clip', 'resnet101_clip', 'resnet50x4_clip', 'resnet50x16_clip', 'resnet50x64_clip', 'vit_b32_clip', 'vit_b16_clip', 'vit_l14_clip', 
                                    'virtex', 'virtex2', 'mlp_mixer', 'mlp_mixer_l16', 
                                    'vit_base_21k', 'vit_large_21k', 'vit_huge', 'deit_base', 
                                    'dino_vitb16', 'dino_resnet50',
                                    'biggan_138k_128size', 'biggan_100k_224size',
                                    'vq_vae_fromScratch', 'soho',
                                    'combinedv1','combinedv2',
                                    'vit_l14_clip_finetune_v2', 'vit_l14_clip_finetune_classAndAtt', 'vit_l14_clip_finetune_class200Epochs', 
                                    'vit_l14_clip_finetune_trainsetAndgenerated_100Epochs', 'vit_l14_clip_finetune_trainsetAndgenerated_200Epochs',
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
                                    'clip_l14_finetun_class_fromMAT_200epochs',
                                    'vit_large_finetune_classes_200epochs']
