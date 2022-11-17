import shlex
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SUN',help='dataset: CUB, AWA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--gen_nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train generater')

parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
parser.add_argument('--beta', type=float, default=1, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dis', type=float, default=3, help='Discriminator weight')
parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')

parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=50, help='training steps of the classifier')

parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=22000)
parser.add_argument('--evl_start',  type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--S_dim', type=int, default=1024)
parser.add_argument('--NS_dim', type=int, default=1024)

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')

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


def CONFIG(dataset="CUB", finetuned_features=False, feature_backbone="resnet101", gpu=0):
    if dataset == "CUB":
        if finetuned_features:
            argString = f'--dataset CUB  --ga 1 --beta 0.003 --dis 0.3 --nSample 1200 --gpu {gpu} --S_dim 2048 --NS_dim 2048 \
                        --lr 0.00003  --classifier_lr 0.002 --gen_nepoch 600 --kl_warmup 0.001 --tc_warmup 0.0001 --weight_decay 1e-8 \
                        --vae_dec_drop 0.1 --dis_step 3 --ae_drop 0.0 --finetune true --feature_backbone {feature_backbone}'
        else:
            argString = f'--dataset CUB  --ga 5 --beta 0.003 --dis 0.3 --nSample 1000 --gpu {gpu} --S_dim 2048 --NS_dim 2048 \
                        --lr 0.0001  --classifier_lr 0.002 --gen_nepoch 600 --kl_warmup 0.001 --tc_warmup 0.0001 --weight_decay 1e-8 \
                        --vae_enc_drop 0.1 --vae_dec_drop 0.1 --dis_step 3 --ae_drop 0.0 --feature_backbone {feature_backbone}'

    elif dataset == "SUN":
        if finetuned_features:
            argString = f'--dataset SUN --ga 30 --beta 0.3 --dis 0.5 --nSample 400 --gpu {gpu} --S_dim 2048 --NS_dim 2048 --lr 0.0003 \
                         --kl_warmup 0.001 --tc_warmup 0.0003 --vae_dec_drop 0.2 --dis_step 3 --ae_drop 0.4 --feature_backbone {feature_backbone}'
        else:
            argString = f'--dataset SUN --ga 3 --beta 0.1 --dis 0.5 --nSample 400 --gpu {gpu} --S_dim 2048 --NS_dim 2048 --lr 0.0003 \
                         --finetune True --feature_backbone {feature_backbone}'

    elif dataset == "AWA2":
        if finetuned_features:
            argString = f'--dataset AWA2 --ga 0.5 --beta 1 --dis 0.3 --nSample 5000 --gpu {gpu} --S_dim 1024 --NS_dim 1024 --lr 0.00003 \
                         --classifier_lr 0.003 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 \
                         --ae_drop 0.2 --gen_nepoch 220 --evl_start 40000 --evl_interval 400 --manualSeed 6152 --feature_backbone {feature_backbone}'
        else:
            argString = f'--dataset AWA2 --ga 0.5 --beta 1 --dis 0.3 --nSample 1800 --gpu {gpu} --S_dim 312 --NS_dim 312 --lr 0.00003 \
                         --classifier_lr 0.0015 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 \
                         --ae_drop 0.2 --gen_nepoch 150 --evl_start 20000 --evl_interval 300 --manualSeed 6152 --finetune true \
                         --weight_decay 3e-7 --classifier_steps 20 --feature_backbone {feature_backbone}'

    args = parser.parse_args(shlex.split(argString))
    return args