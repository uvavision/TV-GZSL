#######################
#author: Shiming Chen
#FREE
#######################
import shlex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB, FLO')
parser.add_argument('--dataroot', default='/vislang/paola/ZERO_SHOT/CADA-VAE-PyTorch/data/SDGZSL_data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--cls_nepoch', type=int, default=2000, help='number of epochs to train for classifier')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument("--conditional", action='store_true',default=True)
###

parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
parser.add_argument('--loop', type=int, default=2)
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
#############################################################
parser.add_argument('--result_root', type=str, default='result', help='root path for saving checkpoint')
parser.add_argument('--center_margin', type=float, default=150, help='the margin in the center loss')
parser.add_argument('--center_weight', type=float, default=0.5, help='the weight for the center loss')
parser.add_argument('--incenter_weight', type=float, default=0.5, help='the weight for the center loss')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
parser.add_argument('--latensize', type=int, default=2048, help='size of semantic features')
parser.add_argument('--i_c', type=float, default=0.1, help='information constrain')
parser.add_argument('--lr_dec', action='store_true', default=False, help='enable lr decay or not')
parser.add_argument('--lr_dec_ep', type=int, default=1, help='lr decay for every 100 epoch')

parser.add_argument('--feature_backbone', default='resnet101',
                        choices=['resnet101', 'CLIP', 'resnet152', 'resnet50', 'resnet50_moco', 'googlenet', 'vgg16', 'alexnet', 
                                 'shufflenet', 'vit', 'vit_large', 'adv_inception_v3', 'inception_v3', 
                                 'resnet50_clip', 'resnet101_clip', 'resnet50x4_clip', 'resnet50x16_clip', 'resnet50x64_clip', 'vit_b32_clip', 'vit_b16_clip', 'vit_l14_clip', 
                                 'virtex', 'virtex2', 'mlp_mixer', 'mlp_mixer_l16', 
                                 'vit_base_21k', 'vit_large_21k', 'vit_huge', 'deit_base', 
                                 'dino_vitb16', 'dino_resnet50',
                                 'biggan_138k_128size', 'biggan_100k_224size',
                                 'vq_vae_fromScratch', 'soho',
                                 'vit_l14_clip_finetune_v2', 'vit_l14_clip_finetune_classAndAtt', 'vit_l14_clip_finetune_class200Epochs', 
                                 'vit_l14_clip_finetune_trainsetAndgenerated_100Epochs', 'vit_l14_clip_finetune_trainsetAndgenerated_200Epochs',
                                 'vit_l14_clip_finetuned_classAndAtt_200Epochs', 
                                 'vit_l14_clip_finetuned_setAndgenerated_classAndAtt_100Epochs', 'vit_l14_clip_finetuned_setAndgenerated_classAndAtt_200Epochs',
                                 'clip_l14_finetun_atts_fromMAT_200epochs',
                                 'clip_l14_finetun_classAndatts_fromMAT_200epochs',
                                 'clip_l14_finetun_class_fromMAT_200epochs',
                                 'vit_large_finetune_classes_200epochs'],
                        help='select feature backbone from resnet101, CLIP, resnet152, resnet50, resnet50_moco, googlenet, vgg16, alexnet, shufflenet, vit, vit_large, adv_inception_v3, inception_v3, virtex, virtex2, mlp_mixer, mlp_mixer_l16, vit_base_21k, vit_large_21k, vit_huge, deit_base, dino_vitb16, dino_resnet50, biggan_138k_128size, biggan_100k_224size, vq_vae_fromScratch, soho')
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')

#################################################################
def CONFIG(dataset="CUB", finetuned_features=False, feature_backbone="resnet101", gpu=0):
    if dataset == "CUB":
        argString = f'--gammaD 10 --gammaG 10 \
                        --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
                        --nepoch 501 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot data --dataset CUB \
                        --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
                        --nclass_all 200 --nclass_seen 150 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048  \
                        --syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8 --feature_backbone {feature_backbone}'
        if finetuned_features:
            argString += " --finetune "

    elif dataset == "SUN":
        argString = f'--gammaD 1 --gammaG 1 --gzsl \
                        --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 601 --ngh 4096 \
                        --a1 0.1 --a2 0.01 --loop 2 --feed_lr 0.0001 \
                        --ndh 4096 --lambda1 10 --critic_iter 1 --dataset SUN \
                        --batch_size 512 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.0002 \
                        --classifier_lr 0.0005 --nclass_seen 645 --nclass_all 717 --dataroot data \
                        --syn_num 300 --center_margin 120 --incenter_weight 0.8 --center_weight 0.5 --recons_weight 0.1 --feature_backbone {feature_backbone}'
        if finetuned_features:
            argString += " --finetune "

    elif dataset == "AWA2":
        argString = f'--gammaD 10 \
                        --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
                        --class_embedding att --nepoch 401 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 1 \
                        --feed_lr 0.0001 --dec_lr 0.0001 --loop 2 --a1 0.01 --a2 0.01 \
                        --nclass_all 50 --dataroot data --dataset AWA2 \
                        --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
                        --lr 0.00001 --classifier_lr 0.001 --nclass_seen 40 --freeze_dec \
                        --syn_num 4600 --center_margin 50 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.5 --feature_backbone {feature_backbone}'
        if finetuned_features:
            argString += " --finetune "

    opt = parser.parse_args(shlex.split(argString))
    opt.lambda2 = opt.lambda1

    if opt.feature_backbone in ['resnet101', 'resnet152', 'resnet50', 'resnet50_moco', 'adv_inception_v3', 'inception_v3', 'virtex', 'virtex2', 'dino_resnet50']:
        image_feature_size = 2048
    elif opt.feature_backbone in ['googlenet', 'shufflenet', 'vit_large', 'resnet50_clip', 'resnet50x64_clip', 'mlp_mixer_l16', 'vit_large_21k']:
        image_feature_size = 1024
    elif opt.feature_backbone in ['vgg16', 'alexnet']:
        image_feature_size = 4096
    elif opt.feature_backbone in ['CLIP', 'resnet101_clip', 'vit_b32_clip', 'vit_b16_clip', 'vq_vae_fromScratch']: ## This is CLIP
        image_feature_size = 512
    elif opt.feature_backbone in ['vit', 'resnet50x16_clip', 'vit_l14_clip', 'mlp_mixer', 'vit_base_21k', 'deit_base', 'dino_vitb16', 'soho',
                                'vit_l14_clip_finetune_v2', 'vit_l14_clip_finetune_classAndAtt', 'vit_l14_clip_finetune_class200Epochs', 
                                'vit_l14_clip_finetune_trainsetAndgenerated_100Epochs', 'vit_l14_clip_finetune_trainsetAndgenerated_200Epochs',
                                'vit_l14_clip_finetuned_classAndAtt_200Epochs', 
                                'vit_l14_clip_finetuned_setAndgenerated_classAndAtt_100Epochs', 'vit_l14_clip_finetuned_setAndgenerated_classAndAtt_200Epochs',
                                'clip_l14_finetun_atts_fromMAT_200epochs',
                                'clip_l14_finetun_classAndatts_fromMAT_200epochs',
                                'clip_l14_finetun_class_fromMAT_200epochs']:
        image_feature_size = 768
    elif opt.feature_backbone in ['resnet50x4_clip']:
        image_feature_size = 640
    elif opt.feature_backbone in ['vit_huge', 'vit_large_finetune_classes_200epochs']:
        image_feature_size = 1280
    elif opt.feature_backbone in ['biggan_138k_128size', 'biggan_100k_224size']:
        image_feature_size = 1536
    opt.resSize = image_feature_size

    if opt.dataset == 'CUB':
        text_feature_size = 312
    elif opt.dataset == 'SUN':
        text_feature_size = 102
    elif opt.dataset == 'AWA2':
        text_feature_size = 85
    opt.attSize = text_feature_size

    opt.encoder_layer_sizes[0] = opt.resSize
    opt.decoder_layer_sizes[-1] = opt.resSize
    opt.latent_size = opt.attSize

    return opt
