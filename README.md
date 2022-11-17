<div align="center">
  <img src="graphics/TV_GZSL.png" alt="TV-GZSL Framework" width="67%"/>
</div>
<h2 align="center">A <b>T</b>oolkit for large scale analysis of <b>V</b>isual features and<br> Generalized Zero-Shot Learning (<b>GZSL</b>) methods</h2>

<hr>

### Table of Contents

* [Requirements](https://github.com/uvavision/TV-GZSL#requirements)
* [Data Setup](https://github.com/uvavision/TV-GZSL#data-setup)
* [FAQ](https://github.com/uvavision/TV-GZSL#faq)
  * [Available Parameters](https://github.com/uvavision/TV-GZSL#available-parameters)
  * [Original Method Repositories](https://github.com/uvavision/TV-GZSL#original-method-repositories---please-cite-all-of-them-accordingly)
  * [Features and Methods](https://github.com/uvavision/TV-GZSL#features-and-methods)
  * [Finetuning](https://github.com/uvavision/TV-GZSL#finetuning)
* [How To: Adding New Methods](https://github.com/uvavision/TV-GZSL#how-to-adding-new-methods)
* [Updates](https://github.com/uvavision/TV-GZSL#updates)

#### Requirements
- python >= 3.7.7 
- pytorch > 1.5.0
- torchvision
- tensorflow-gpu==1.14
- torchcontrib

#### Data Setup
1. Clone repository and create a new `data` directory 
    ```
    ~$ git clone https://github.com/uvavision/TV-GZSL.git
    ~$ cd TV-GZSL
    ~$ mkdir data
    ```
2. Download all files located in [this folder](https://drive.google.com/drive/folders/14NQE2px2GPh6aucMk6aPfuiikdiSGduI?usp=sharing). 
   - You can also download only the features from the backbones you want to use in your experiments.
   - If you want to use the traditional RN101 and RN101 fine-tuned features for each dataset you can download [this folder](https://drive.google.com/drive/folders/18FtMt2R1BOlvdFYggxuh0KE21ElIen07) only. Just make sure it is inside the `data` directory you just created.

#### Features and Methods

| Datasets | Backbone Types | GZSL Families |
| :-----------: | :-----------: | :-----------: |
| CUB | CNN | Embedding-based |
| SUN | ViT | Generative-based |
| AWA2 | MLP-Mixer | Disentanglement-based |

#### FAQ
- Code is based on original authors implementations, including seed and hyperparameter selection.
- Codebase should be used to reproduce the results we report.
- Run the command below to reproduce the CADA-VAE results on CUB using the RN101 features:
```
CUDA_VISIBLE_DEVICES=0 python main.py --method CADA --dataset CUB --feature_backbone resnet101
```
- If you want to use the fine-tuned features you should add the finetuned_features flag:
```
CUDA_VISIBLE_DEVICES=0 python main.py --method CADA --dataset CUB --feature_backbone resnet101 --finetuned_features
```
- If you want to use the a different method and feature you should add the feature_backbone flag and change the method name:
  - Method name: ```--method SDGZSL```
  - Use CLIP w/ ViT/B32 features: ```--feature_backbone vit_b32_clip```
  - Run your code in a different GPU: ```CUDA_VISIBLE_DEVICES=1``` 
```
CUDA_VISIBLE_DEVICES=1 python main.py --method SDGZSL --dataset CUB --feature_backbone vit_b32_clip
```

#### Available Parameters

Everything you need to run is in main.py.
The Wrapper class contains all the main functions to create the model, prepare the dataset, and train your model. The arguments you pass are handled by the Wrapper.<br>
Please play a special attention to the ```--feature_backbone``` parameter to use the pre-computed features you are looking for!

```python
usage: main.py [-h] [--dataset DATASET]
               [--feature_backbone {resnet101,resnet152,resnet50,resnet50_moco,googlenet,vgg16,alexnet,shufflenet,vit,vit_large,adv_inception_v3,inception_v3,resnet50_clip,resnet101_clip,resnet50x4_clip,resnet50x16_clip,resnet50x64_clip,vit_b32_clip,vit_b16_clip,vit_l14_clip,virtex,virtex2,mlp_mixer,mlp_mixer_l16,vit_base_21k,vit_large_21k,vit_huge,deit_base,dino_vitb16,dino_resnet50,biggan_138k_128size,biggan_100k_224size,vq_vae_fromScratch,soho,combinedv1,combinedv2,vit_l14_clip_finetune_v2,vit_l14_clip_finetune_classAndAtt,vit_l14_clip_finetune_class200Epochs,vit_l14_clip_finetune_trainsetAndgenerated_100Epochs,vit_l14_clip_finetune_trainsetAndgenerated_200Epochs,vit_l14_clip_finetuned_classAndAtt_200Epochs,vit_l14_clip_finetuned_setAndgenerated_classAndAtt_100Epochs,vit_l14_clip_finetuned_setAndgenerated_classAndAtt_200Epochs,clip_l14_finetune_classes_200epochs,clip_l14_finetun_atts_200epochs,clip_l14_finetun_atts_200epochs,clip_l14_finetune_classes_200epochs_frozenAllExc1Layer,clip_l14_finetun_atts_200epochs_frozenAllExc1Layer,clip_l14_finetune_classAndAtt_200epochs_frozenAllExc1Layer,clip_l14_finetune_classes_200epochs_frozenTextE,clip_l14_finetun_atts_200epochs_frozenTextE,clip_l14_finetune_classAndAtt_200epochs_frozenTextE,clip_l14_finetun_atts_fromMAT_200epochs,clip_l14_finetun_classAndatts_fromMAT_200epochs,clip_l14_finetun_class_fromMAT_200epochs,vit_large_finetune_classes_200epochs}]
               [--methods {DEVISE,ESZSL,ALE,CADA,tfVAEGAN,CE,SDGZSL,FREE,UPPER_BOUND}]
               [--finetuned_features] [--data_path DATA_PATH]
               [--workers WORKERS] [--dropout DO] [--optimizer OPTIMIZER]
               [--epochs N] [--start_epoch N] [-b N] [--lr LR]
               [--initial_lr LR] [--lr_rampup EPOCHS]
               [--lr_rampdown_epochs EPOCHS] [--momentum M] [--nesterov]
               [--weight-decay W] [--doParallel] [--print_freq N]
               [--root_dir ROOT_DIR] [--add_name ADD_NAME] [--exp_dir EXP_DIR]
               [--load_from_epoch LOAD_FROM_EPOCH] [--seed SEED]
```

<!-- <br/> -->


#### Original Method Repositories: - please cite all of them accordingly!
- [[ICCV2021] Official Pytorch implementation for **SDGZSL**: Semantics Disentangling for Generalized Zero-Shot Learning](https://github.com/uqzhichen/SDGZSL) 
- [[ICCV2021] Official Pytorch implementation for **FREE**: Feature Refinement for Generalized Zero-Shot Learning](https://github.com/shiming-chen/FREE)
- [[CVPR2021] Official Pytorch implementation for **CE**: Contrastive Embedding for Generalized Zero-Shot Learning](https://github.com/Hanzy1996/CE-GZSL)
- [[ECCV 2020] Official Pytorch implementation for **tfVAEGAN**: "Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification"](https://github.com/akshitac8/tfvaegan)
- [[CVPR2019] Official Pytorch implementation for **CADA-VAE**: "Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders"](https://github.com/edgarschnfld/CADA-VAE-PyTorch)
- [[2020] **ALE** (2016), **DEVISE** (2013) and **ESZSL** (2015) implementations from *Soumava Paul - https://mvp18.github.io/* - github repository](https://github.com/mvp18/Popular-ZSL-Algorithms)


#### Finetuning
1. Download the dataset images and annotations:
   1. CUB: http://www.vision.caltech.edu/datasets/cub_200_2011/
   2. SUN: https://cs.brown.edu/~gmpatter/sunattributes.html
   3. AWA2: https://cvml.ist.ac.at/AwA2/
2. Unzip them in a ```data``` folder inside the ```finetune``` folder:
    ```
    ~$ cd anonymized_code/finetune/
    ~$ mkdir data
    ~$ tar -xvf [filename]
    ```
3. Finetune:
   - Dataloaders: [sample code](https://github.com/uvavision/TV-GZSL/blob/main/finetune/dataloader.py)
   - Unimodal Backbones: [sample code](https://github.com/uvavision/TV-GZSL/blob/main/finetune/unimodal/vit_finetune.py)
   - CLIP: [sample code](https://github.com/uvavision/TV-GZSL/blob/main/finetune/multimodal/clip_finetune.py)


#### How to: Adding New Methods

You can add a new method under the ```methods``` folder. Then, you should only modify the ```utils/general_config.py``` and ```wrapper.py``` files to reference your new method:

1. Add your method name in the choices array of the methods argument in ```utils/general_config.py``` array ``` all_methods```.
2. In ```wrapper.py``` you should include the new parameter option when initializing the ```Wrapper Class```. 
3. To support all available features in your custom method: ```from utils.cada_dataloader import DATA_LOADER``` 
4. To reuse the final classifier for Generative-based and Disentanglement-based methods, you can use the ```LINEAR_LOGSOFTMAX``` class inside ```wrapper.py```

_______


#### Updates

- :white_check_mark: All 54 visual features for all datasets are [available here!](https://drive.google.com/drive/folders/14NQE2px2GPh6aucMk6aPfuiikdiSGduI?usp=sharing) :star: 
- :white_check_mark: Initial codebase is now available! :arrow_double_up:
- :black_square_button: Please expect regular updates and commits of this repo.

<br><br>

<hr>


## On the **T**ransferability of **V**isual Features in Generalized Zero-Shot Learning (**GZSL**) :: **TV-GZSL**
### About
Our work provides a comprehensive benchmark for Generalized Zero-Shot Learning (GZSL). We benchmark extensively the utility of different GZSL methods which we characterize as embedding-based, generative-based, and based on semantic disentanglement. We particularly investigate how these previous methods for GZSL fare against CLIP, a more recent large scale pretrained model that claims zero-shot performance by means of being trained with internet scale multimodal data. Our findings indicate that through prompt engineering over an off-the-shelf CLIP model, it is possible to surpass all previous methods on standard benchmarks for GZSL: CUB (Birds), SUN (scenes), and AWA2 (animals). While it is possible that CLIP has actually seen many of the unseen categories in these benchmarks, we also show that GZSL methods in combination with the feature backbones obtained through CLIP contrastive pretraining (e.g. ViT~L/14) still provide advantages in standard GZSL benchmarks over off-the-shelf CLIP with prompt engineering. In summary, some GZSL methods designed to transfer information from seen categories to unseen categories still provide valuable gains when paired with a comparable feature backbone such as the one in CLIP. Surprisingly, we find that generative-based GZSL methods provide more advantages compared to more recent methods based on semantic disentanglement. We release a well-documented codebase which both replicates our findings and provides a modular framework for analyzing representation learning issues in GZSL.