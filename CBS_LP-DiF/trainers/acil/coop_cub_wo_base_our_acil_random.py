import os.path as osp
from statistics import mode

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset as TorchDataset
from dassl.data.transforms.transforms import build_transform
from ..coop import load_clip_to_cpu, PromptLearner, TextEncoder
from tqdm import tqdm
import random
import scipy.io as sio
import os
from PIL import Image
import pickle
from utils import calculate_GD
import time
_tokenizer = _Tokenizer()



def read_file_to_list(file_name):

    with open(file_name, 'r') as file:
        lines = [line.strip() for line in file]

    return lines



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # self.MLP = nn.sequence(
        #     nn.Linear(512,512),
        #     nn.ReLU(True)
        # )

    def forward(self, image, pseudo_feat=None, with_emb=False):


        image_features = self.image_encoder(image.type(self.dtype))

        if with_emb:
            emb = image_features
        # exit()

        prompts = self.prompt_learner()
        # print(prompts)
        tokenized_prompts = self.tokenized_prompts
        # print(tokenized_prompts)
        text_features = self.text_encoder(prompts, tokenized_prompts)
    

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        logits = logit_scale * image_features @ text_features.t()

        if with_emb:
            return logits, emb
        if type(pseudo_feat)!=type(None):
            pseudo_feat = pseudo_feat.half()
            # image_features = torch.cat((image_features,pseudo_feat))
        
            logits_pseudo = logit_scale * pseudo_feat @ text_features.t()
            return logits,logits_pseudo
        
        return logits
    
    def forward_pseudo(self, pseudo_feat):
        # image_features = self.image_encoder(image.type(self.dtype))
        # exit()

        prompts = self.prompt_learner()
        # print(prompts)
        tokenized_prompts = self.tokenized_prompts
        # print(tokenized_prompts)
        text_features = self.text_encoder(prompts, tokenized_prompts)

    
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * pseudo_feat @ text_features.t()

        return logits


class CUB200_FEW(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode,class_per_task,test_model='all') -> None:

        # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)

        # with open("cifar100_train.pkl",'wb') as f:
        #     pickle.dump(cifar100,f,-1)
        # exit()
        self.tfm = tfm
        self.mode = mode

        self.name = 'cub200'

            
        image_data_txt = 'data/CUB_200_2011/images.txt'

        image_root = 'data/CUB_200_2011/images'

        label_txt = 'data/CUB_200_2011/image_class_labels.txt'


        train_test_split = 'data/CUB_200_2011/train_test_split.txt'

        self.class_per_task = class_per_task
        # self.novel_task_len = int(200/class_per_task)
        self.task_len = int(200/class_per_task)
        self.task_id = task_id

        image_id_split = {}

        with open(train_test_split,'r') as f:
            image_split = f.readlines()
            for i in range(len(image_split)):
                image_split[i] = image_split[i].replace('\n','')
                image_id,is_train = image_split[i].split(" ")
                image_id_split[image_id] = eval(is_train)

        image_id_path_dict = {}

        with open(image_data_txt,'r') as f:
            image_id_list = f.readlines()
            for i in range(len(image_id_list)):
                image_id_list[i] = image_id_list[i].replace('\n','')
                image_id,path = image_id_list[i].split(" ")
                image_id_path_dict[image_id] = os.path.join(image_root,path)

        image_id_label_dict = {}

        with open(label_txt,'r') as f:
            image_label_list = f.readlines()
            for i in range(len(image_label_list)):
                image_label_list[i] = image_label_list[i].replace('\n','')
                image_id,label = image_label_list[i].split(" ")
                image_id_label_dict[image_id] = eval(label)-1

        self.images_list = []
        self.labeled_list = []



        if mode=='train':

            task_split = [[] for x in range(self.task_len)]



            for i in range(self.task_len):
                for j in range(self.class_per_task):
                    task_split[i].append(i*self.class_per_task+j)

            select_class_id = task_split[task_id]
            self.end_class_id = select_class_id[-1]
           
            self.class_idx_dict = {x:[] for x in select_class_id}

            print(select_class_id)
            for key in image_id_path_dict:
                if image_id_split[key]==1:
                    label = image_id_label_dict[key]
                    if label in  self.class_idx_dict:
                        self.class_idx_dict[label].append(key)


            for c in select_class_id:
                idx_list = self.class_idx_dict[c]

                for id in idx_list:
                    self.images_list.append(image_id_path_dict[id])
                    self.labeled_list.append(image_id_label_dict[id])

            

            self.len = len(self.images_list)

            self.idxs_lb = np.zeros(self.len, dtype=bool)
        else:
            if test_model=='all':
                self.data = []
                task_to_id_end = {0:class_per_task}
                start = self.class_per_task+self.class_per_task
                for i in range(1,self.task_len):
                    task_to_id_end[i]=start
                    start+=self.class_per_task

                select_class_id=[x for x in range(task_to_id_end[task_id])]

                print(select_class_id)

                for key in image_id_path_dict:
                    if image_id_split[key]==0:
                        label = image_id_label_dict[key]
                        if label in  select_class_id:
                            self.images_list.append(image_id_path_dict[key])
                            self.labeled_list.append(label)


                self.shot = shot

                self.len = len(self.images_list)
            else:
                task_split = [[] for x in range(9)]

                for i in range(60):
                    task_split[0].append(i)

                for i in range(1,9):
                    for j in range(5):
                        task_split[i].append(60+(i-1)*5+j)
                # self.class_idx_dict = 

                # print(select_class_id)
                select_class_id = task_split[task_id]

                # print(select_class_id)


                cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=False,transform=tfm)

                self.class_idx_dict = {x:[] for x in select_class_id}
                for i in range(len(cifar100)):
                    image,label = cifar100[i]

                    if label in select_class_id:

                        self.data.append(cifar100[i])
                self.len = len(self.data)


    

    def __getitem__(self,idx):

        img_name = self.images_list[idx]
        label = self.labeled_list[idx]
        # img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.tfm:
            image = self.tfm(image)
        return image,label



    def __len__(self):
        return self.len
    



class GDDataset(TorchDataset):

    def __init__(self, tfm, task_id,save_dir) -> None:


        self.tfm = tfm

        self.task_id = task_id
        


        pre_root = os.path.join(save_dir,'session_{}.txt'.format(task_id))
        # pre_root = 'CUB_200_2011/ACIL_entropy/R_5/SEED0/
        old_images_list = []
        old_label_list = []
        # self.shot = shot
        lines = read_file_to_list(pre_root)

        for line in lines:
            image_path, image_label = line.split(" ")
            old_images_list.append(image_path)
            old_label_list.append(int(image_label))

        
        self.images_list = old_images_list
        self.labeled_list = old_label_list

        self.len = len(self.images_list)

    

    def __getitem__(self,idx):

        img_name = self.images_list[idx]
        label = self.labeled_list[idx]
        # img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.tfm:
            image = self.tfm(image)
        return image,label


    def __len__(self):
        return self.len




@TRAINER_REGISTRY.register()
class CoOp_CUB_wo_Base_Our_ACIL_Random(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]




    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        task_id_now = self.cfg.TRAINER.TASK_ID

        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)
        # if self.eval_only==False:
        self.tfm_train = build_transform(self.cfg, is_train=True)

        train_set_task0 = CUB200_FEW(shot=self.cfg.DATASET.NUM_SHOTS,tfm=self.tfm_train,class_per_task=20,task_id = task_id_now,mode='train')

        self.train_set = train_set_task0
        
        train_loader = torch.utils.data.DataLoader(train_set_task0,batch_size=50,num_workers=4,drop_last=False,shuffle=True)

        self.train_loader_x = train_loader

        self.tfm_test = build_transform(self.cfg, is_train=False)
        test_set_task0 = CUB200_FEW(shot=self.cfg.DATASET.NUM_SHOTS,tfm=self.tfm_test,class_per_task=20,task_id = task_id_now,mode='test',test_model='all')

        
        test_loader = torch.utils.data.DataLoader(test_set_task0,batch_size=100,num_workers=4,drop_last=False)

  

        
        # self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = test_loader  # optional, can be None
        self.test_loader = test_loader

        self.num_classes = 200
        self.classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']

        # self.num_source_domains = dm.num_source_domains
        self.lab2cname = {x:self.classnames[x] for x in range(200)}  # dict {label: classname}

        # self.dm = dm

    def build_model(self):
        cfg = self.cfg
        task_id_now = self.cfg.TRAINER.TASK_ID
        self.task_id = self.cfg.TRAINER.TASK_ID
        class_per_task = self.cfg.TRAINER.CLASS_PER_TASK
        self.class_per_task = class_per_task
        task_num = int(200/class_per_task)
        self.task_end_id= {0:class_per_task}
        start = class_per_task+class_per_task
        for i in range(1,task_num):
            self.task_end_id[i]=start
            start+=class_per_task
        # self.task_end_id = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        # self.class_id_to_task_id = {}

        # for i in range(200):
        #     if i<100:
        #         self.class_id_to_task_id[i] = 0
        #     else:
        #         self.class_id_to_task_id[i] = ((i-100)//10)+1

        self.class_end = self.task_end_id[task_id_now]
        self.classnames_all = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        class_task = self.classnames_all[:self.class_end]
        # class_task = self.classnames_all


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        self.clip_model = clip_model
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, class_task, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model,device_ids=[0, 1, 2,3,4])



    def before_round(self):
        import time
        from dassl.utils import mkdir_if_missing

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_round(self):
        import time, datetime
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
    

    def write_selected_sample(self,active_alg,task_id,save_path):

        save_file = os.path.join(save_path,'session_{}.txt'.format(task_id))

        image_path_list = [active_alg.dset.images_list[ind] for ind in np.where(active_alg.dset.idxs_lb)[0].tolist()]
        label_list = [active_alg.dset.labeled_list[ind] for ind in np.where(active_alg.dset.idxs_lb)[0].tolist()]
        with open(save_file,'w') as f:
            for i in range(len(image_path_list)):
                f.write(image_path_list[i]+" "+str(label_list[i])+"\n")

        return

    def save_GD(self,GD,save_path):
        save_file = os.path.join(save_path,'GD.pkl')


        with open(save_file,'wb') as f:
            pickle.dump(GD,f,-1)
        return

    def update_GD(self,GD,save_path):
        save_file = os.path.join(save_path,'GD.pkl')

        with open(save_file,'rb') as f:
            old_GD = pickle.load(f)

        GD.update(old_GD)

        return GD

        

    def train(self):
        
        from active import get_strategy
        ROUND = self.cfg.AL.ROUND
        select_data_save_path = "data/"+self.cfg.AL.SAMPLE_SAVE_PATH

        if not os.path.exists(select_data_save_path):
            os.makedirs(select_data_save_path)



        self.before_train()


        budget = self.class_per_task
        al_method = self.cfg.AL.NAME # 默认第一回合都是random
        cl_method = self.cfg.CL.NAME
        assert hasattr(self.train_set, 'idxs_lb'), "The dataset is not modified for Active learning."
        active_alg = get_strategy(al_method, self.train_set, self.model, self.device, self.cfg, self.tfm_train,self.tfm_test, select_data_save_path, cl_method)

        print("begin actively select sample in round.")
        idxs_active = active_alg.query(budget*ROUND)
        active_alg.update(idxs_active)
        self.train_set.idxs_lb = active_alg.dset.idxs_lb  # 其实应该是不需要的，为了以防万一，更新一下

        
        train_loader_x = active_alg.build_label_loaders()
        # self.dm.train_loader_x = train_loader_x
        self.train_loader_x = train_loader_x
        print("训练Loader长度:", len(train_loader_x))

                # 测试用，跳过第一回合
                # continue
            
            
            # 执行训练
        self.start_epoch = 0
        self.max_epoch = self.cfg.OPTIM.MAX_EPOCH
        print("Begin round  of task {}.".format(self.task_id))
        self.write_selected_sample(active_alg,self.task_id, select_data_save_path)
        # if R==ROUND-1:
        self.max_epoch = self.cfg.OPTIM.MAX_EPOCH
        
        GD = self.generate_GD(select_data_save_path)
        if self.task_id>0:
            GD = self.update_GD(GD,select_data_save_path)
        self.save_GD(GD, select_data_save_path)


        # if R != 0:
        #     self.before_round()
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
            start_time = time.time()
            self.before_epoch()
            self.run_epoch()
            end_time = time.time()
            print("runing time: ", end_time - start_time)
            self.after_epoch()
            

            
        
        # if R != ROUND - 1:
        #     self.after_round()

        
        self.after_train()



    def model_inference_feats(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        return image_features

    def generate_GD(self,select_data_save_path):

        GD_set = GDDataset(self.tfm_test,self.task_id,select_data_save_path)

        GD_loader = torch.utils.data.DataLoader(GD_set,batch_size=1,num_workers=1,drop_last=False)

        # GD = {}

        feats_label = []

        for idx, input in enumerate(tqdm(GD_loader)):
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
    
            output = self.model_inference_feats(image)

            image_features = output.cpu()

            label = label.cpu()

            data_point = {'feats':image_features.numpy(),'label':label.numpy()}
            feats_label.append(data_point)


        begin_class = self.task_id * self.class_per_task

        GD = calculate_GD(feats_label, begin_class,self.class_per_task)

        return GD


    def forward_backward(self, batch):

        
        # if self.task_id == 0:
        # image, label = batch
        # image = image.to(self.device)
        # label = label.to(self.device)
        
        # prec = self.cfg.TRAINER.COOP.PREC
        # if prec == "amp":
        #     with autocast():
        #         output = self.model(image)
        #         loss = F.cross_entropy(output, label)
        #     self.optim.zero_grad()
        #     self.scaler.scale(loss).backward()
        #     self.scaler.step(self.optim)
        #     self.scaler.update()
        # else:
        #     output = self.model(image)
        #     loss = F.cross_entropy(output, label)
        #     self.model_backward_and_update(loss)

        if self.task_id == 0:
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            
            prec = self.cfg.TRAINER.COOP.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)
        else:
            image, label,pseudo_feat,pseudo_label = batch

            pseudo_feat = pseudo_feat.view(-1,pseudo_feat.shape[-1])
            
            pseudo_label = pseudo_label.view(-1)

            image = image.to(self.device)

            label = label.to(self.device)
            pseudo_label = pseudo_label.to(self.device)
            # print(pseudo_label)
            pseudo_feat = pseudo_feat.to(self.device)

            
            prec = self.cfg.TRAINER.COOP.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output,output_pseudo = self.model(image,pseudo_feat)

                # print(output.shape,output_pseudo.shape)

                loss = F.cross_entropy(torch.cat((output,output_pseudo)), torch.cat((label,pseudo_label)))
                self.model_backward_and_update(loss)



   
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        ans = 0
        # correct = 0
        # total = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = batch
            # predict_task = []
            # label_task = []
            # for l in label.numpy():
            #     label_task.append(self.class_id_to_task_id[l])
            
            
            image = image.to(self.device)
            label = label.to(self.device)
         
    
            output = self.model_inference(image)

            # pred = output.max(1)[1].cpu().numpy()
            
            # for p in pred:
            #     predict_task.append(self.class_id_to_task_id[p])

            # gt= torch.Tensor(label_task)
            # pred = torch.Tensor(predict_task)
            # matches = pred.eq(gt).float()
            # correct += int(matches.sum().item())
            # total += gt.shape[0]
            # task_mask = torch.zeros([output.shape[0],output.shape[1]])

            # for i in range(label.shape[0]):
            #     label_task = self.class_id_to_task_id[label.cpu().numpy()[i]]
            #     if label_task==0:
            #         start = 0
            #         end=100
            #     else:
            #         start = 100+(label_task-1)*10
            #         end = 100+label_task*10
            #     for j in range(start,end):
            #         task_mask[i][j]=1
            # task_mask = task_mask.to(output.device)


            # output = output*task_mask

            
            self.evaluator.process(output, label)
       
        # print(correct/total)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)



    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        # with torch.no_grad():
        #     prompts = self.model.module.prompt_learner()
        #     # print(prompts)
        #     tokenized_prompts = self.model.module.tokenized_prompts
        #     # print(tokenized_prompts)
        #     text_features = self.model.module.text_encoder(prompts, tokenized_prompts)
            
        #     text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).cpu().numpy()
        # with open(os.path.join(self.output_dir,"training_text_encoding_epoch_{}.pkl".format(self.epoch)),'wb') as f:
        #     pickle.dump(text_features[100:],f,-1)



        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)




