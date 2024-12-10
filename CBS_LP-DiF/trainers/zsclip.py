import torch
import torch.nn as nn
import os
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms.transforms import build_transform

from clip import clip
from torchvision.datasets import CIFAR100
import random
from clip.model import convert_weights
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from tqdm import tqdm
from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
import pickle
import os.path as osp
import numpy as np
import json

def read_file_to_list(file_name):

    with open(file_name, 'r') as file:
        lines = [line.strip() for line in file]

    return lines


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "CIFAR100_full": "a photo of a {}.",
    "CUB200_few": "a photo of a {}.",
    "SlimageNet": "a photo of a {}.",
    "AWA2": "a photo of a {}.",
    "miniImageNet": "a photo of a {}."
}

templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        all_answer = {}
        for i in range(len(templates)):
            temp = templates[i]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            print(f"Prompts: {prompts}")
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to(self.device)

            with torch.no_grad():
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu()
            all_answer[i] = text_features
        
        # pickle.

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        
        return logits




class CIFAR100_FEW(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode) -> None:

        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        task_id_end = task_split[task_id]


        if mode=='train':
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in range(task_id_end):
                idx_list = self.class_idx_dict[c]
                if c>=60:
                    sample_few_idx = random.sample(idx_list,shot)
                    for id in sample_few_idx:
                        self.data.append(self.cifar100[id])
                else:
                    for id in idx_list:
                        self.data.append(self.cifar100[id])


            self.shot = shot

            
        else:
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=False,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in range(task_id_end):
                idx_list = self.class_idx_dict[c]
                for id in idx_list:
                    self.data.append(self.cifar100[id])

        self.len = len(self.data)

    

    def __getitem__(self,index):
        return self.data[index]


    def __len__(self):
        return self.len


class CIFAR100_FEW_CL(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode) -> None:

        # task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        # task_split = [[] for x in range(10)]
        task_split = [[] for x in range(9)]

        for i in range(60):
            task_split[0].append(i)

        for i in range(1,9):
            for j in range(5):
                task_split[i].append(60+(i-1)*5+j)

        # for i in range(60):
        #     task_split[0].append(i)

        # for i in range(10):
        #     for j in range(10):
        #         task_split[i].append(i*10+j)
        task_id_end = task_split[task_id]


        if mode=='train':
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in task_id_end:
                idx_list = self.class_idx_dict[c]
                # if c>=60:
                #     idx_list = random.sample(idx_list,shot)
                for id in idx_list:
                    self.data.append(self.cifar100[id])
                # else:
                #     for id in idx_list:
                #         self.data.append(self.cifar100[id])


            self.shot = shot

            
        else:
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=False,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in range(task_id_end):
                idx_list = self.class_idx_dict[c]
                for id in idx_list:
                    self.data.append(self.cifar100[id])

        self.len = len(self.data)

    

    def __getitem__(self,index):
        return self.data[index]


    def __len__(self):
        return self.len


@TRAINER_REGISTRY.register()
class ZeroshotCLIP3(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
    


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load("RN50", device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100 = CIFAR100_FEW(tfm_test)
        cifar100 =  CIFAR100_FEW(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=self.cfg.TRAINER.TASK_ID,mode='test')
        
        data_loader = torch.utils.data.DataLoader(cifar100,batch_size=100,num_workers=4)

        # if split is None:
        #     split = self.cfg.TEST.SPLIT

        # if split == "val" and self.val_loader is not None:
        #     data_loader = self.val_loader
        # else:
        #     split = "test"  # in case val_loader is None
        #     data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        ans = 0
        for idx, input in enumerate(tqdm(data_loader)):

            
            # input, label = self.parse_batch_test(batch)
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
            output = self.model_inference(image)
  
 
            
            self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    



class CUB200_FEW_ACIL_RANDOM(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode,test_model='all') -> None:

        # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)

        # with open("cifar100_train.pkl",'wb') as f:
        #     pickle.dump(cifar100,f,-1)
        # exit()
        self.tfm = tfm
        image_data_txt = 'CUB_200_2011/images.txt'

        image_root = 'CUB_200_2011/images'

        label_txt = 'CUB_200_2011/image_class_labels.txt'


        train_test_split = 'CUB_200_2011/train_test_split.txt'

        class_per_task=20
        self.class_per_task = class_per_task
        self.novel_task_len = int(200/class_per_task)
        # self.task_len = self.novel_task_len+1
        self.task_len = self.novel_task_len
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
            ramdom_root = 'CUB_200_2011/ACIL_random/R_5/SEED0/session_{}.txt'
            # self.shot = shot
            for i in range(self.task_id+1):
                lines = read_file_to_list(ramdom_root.format(i))

                for line in lines:
                    image_path, image_label = line.split(" ")
                    self.images_list.append(image_path)
                    self.labeled_list.append(int(image_label))

            self.len = len(self.images_list)
            print(self.len)
        else:
            if test_model=='all':
                self.data = []
                # task_to_id_end = {0:100}
                # start = 110
                # for i in range(1,11):
                #     task_to_id_end[i]=start
                #     start+=10

                task_to_id_end = {0:10}
                start = 10+self.class_per_task
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


                # print(select_class_id)


                # for c in select_class_id:
                #     idx_list = self.class_idx_dict[c]
   
                #     for id in idx_list:
                #         self.images_list.append(image_id_path_dict[id])
                #         self.labeled_list.append(image_id_path_dict[id])

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

                print(select_class_id)


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



class CUB200_FEW(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode,test_model='all') -> None:

        # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)

        # with open("cifar100_train.pkl",'wb') as f:
        #     pickle.dump(cifar100,f,-1)
        # exit()
        self.tfm = tfm
        image_data_txt = 'CUB_200_2011/images.txt'

        image_root = 'CUB_200_2011/images'

        label_txt = 'CUB_200_2011/image_class_labels.txt'


        train_test_split = 'CUB_200_2011/train_test_split.txt'

        class_per_task=20
        self.class_per_task = class_per_task
        self.novel_task_len = int(200/class_per_task)
        # self.task_len = self.novel_task_len+1
        self.task_len = self.novel_task_len
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

        # for i in range(len(image_id_list)):
        #     item = image_id_list[i]
        #     image_name = item.split(" ")[1]
        #     images_list.append(os.path.join(image_root,image_name))

        # with open(image_root,'r') as f:
        #     image_id_list = f.readlines()
        #     for i in range(len(image_id_list)):
        #         image_id_list[i] = image_id_list[i].replace('\n','')

        if mode=='train':


            # task_split = [[] for x in range(20)]

            # task_to_id_end = {0:10}
            # start = 20
            # for i in range(1,11):
            #     task_to_id_end[i]=start
            #     start+=10

            # task_split = [[] for x in range(11)]

            # for i in range(100):
            #     task_split[0].append(i)

            task_split = [[] for x in range(self.task_len)]



            for i in range(self.task_len):
                for j in range(self.class_per_task):
                    task_split[i].append(i*self.class_per_task+j)

            select_class_id = task_split[task_id]
            self.end_class_id = select_class_id[-1]
           
            self.class_idx_dict = {x:[] for x in select_class_id}

            # for i in range(1,11):
            #     for j in range(10):
            #         task_split[i].append(i*10+j)
            # for i in range(1,self.task_len):
            #     for j in range(self.class_per_task):
            #         task_split[i].append(100+(i-1)*self.class_per_task+j)
            # self.class_idx_dict = 
            # select_class_id = task_split[task_id]
            # # print(select_class_id)
            # self.class_idx_dict = {x:[] for x in select_class_id}

            # print(select_class_id)
            for key in image_id_path_dict:
                if image_id_split[key]==1:
                    label = image_id_label_dict[key]
                    if label in  self.class_idx_dict:
                        self.class_idx_dict[label].append(key)
    

            # for i in range(len(cifar100)):
            #     image,label = cifar100[i]

            #     if label in self.class_idx_dict:

            #         self.class_idx_dict[label].append(i)

            # self.data = []
            # print(select_class_id)
            for c in select_class_id:
                idx_list = self.class_idx_dict[c]
                # if c>=100:
                #     idx_list = random.sample(idx_list,shot)
                for id in idx_list:
                    self.images_list.append(image_id_path_dict[id])
                    self.labeled_list.append(image_id_label_dict[id])
       
   
            self.shot = shot

            self.len = len(self.images_list)
        else:
            if test_model=='all':
                self.data = []
                # task_to_id_end = {0:100}
                # start = 110
                # for i in range(1,11):
                #     task_to_id_end[i]=start
                #     start+=10

                task_to_id_end = {0:10}
                start = 10+self.class_per_task
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


                # print(select_class_id)


                # for c in select_class_id:
                #     idx_list = self.class_idx_dict[c]
   
                #     for id in idx_list:
                #         self.images_list.append(image_id_path_dict[id])
                #         self.labeled_list.append(image_id_path_dict[id])

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

                print(select_class_id)


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
    

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_CUB(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        self.classnames = classnames
        # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        class_per_task=10
        task_num = int(200/class_per_task)
        self.task_end_id= {0:10}
        start = 10+class_per_task
        for i in range(1,task_num):
            self.task_end_id[i]=start
            start+=class_per_task
        # task_split = {}
        # for i in range(20):
        #     task_split[i] = (i+1)*10
        self.task_id_end = self.task_end_id[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
    
        # with open("noval_classes_better.json") as f:
        #     gpt_data = json.load(f)
  

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load("ViT-B/16", device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]



        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        # sentences_feat = {}

        # for key in gpt_data:
        #     class_sentences = gpt_data[key]
        #     prompts = torch.cat([clip.tokenize(p) for p in class_sentences])
        #     prompts = prompts.to(self.device)
        #     with torch.no_grad():
        #         text_features = clip_model.encode_text(prompts)
        #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     sentences_feat[key] = text_features.cpu().numpy()
        # with open("novel_class_gpt_sentence_feats.pkl","wb") as f:
        #     pickle.dump(sentences_feat,f,-1)
        # exit()


        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)r
        cfg = self.cfg
        classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        self.classnames = classnames
        # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        task_split = {}
        for i in range(20):
            task_split[i] = (i+1)*10
        self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
        self.lab2cname = {x:self.classnames[x] for x in range(200)}
        return 
        # task_id_now = self.cfg.TRAINER.TASK_ID
        # tfm_train = build_transform(self.cfg, is_train=True)
        # cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        # tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        # train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        # test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        # self.train_loader_x = train_loader
        # # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = test_loader  # optional, can be None
        # self.test_loader = test_loader

        # self.num_classes = 100
        # self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # # self.num_source_domains = dm.num_source_domains
          # dict {label: classname}

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


    def model_inference_feats(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        return image_features



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        # tfm_test = build_transform(self.cfg, is_train=False)
        tfm_test = build_transform(self.cfg, is_train=True)
        # cifar100 = CIFAR100_FEW(tfm_test)
        cub_200 =  CUB200_FEW_ACIL_RANDOM(shot=5,tfm=tfm_test,task_id=self.cfg.TRAINER.TASK_ID,mode='train')
        
        data_loader = torch.utils.data.DataLoader(cub_200,batch_size=1,num_workers=4)

        # if split is None:
        #     split = self.cfg.TEST.SPLIT

        # if split == "val" and self.val_loader is not None:
        #     data_loader = self.val_loader
        # else:
        #     split = "test"  # in case val_loader is None
        #     data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        ans = 0
        id = 0
        # for j in range(4):
        ans_list =[ ]
        for idx, input in enumerate(tqdm(data_loader)):
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
            # output = self.model_inference(image)
  
 
            
            # self.evaluator.process(output, label)

            
            # input, label = self.parse_batch_test(batch)
        #     image = input[0].to(self.device)
        #     label = input[1].to(self.device)
    
            output = self.model_inference_feats(image)

            image_features = output.cpu()

            label = label.cpu()

            data_point = {'feats':image_features.numpy(),'label':label.numpy()}

            save_path = 'extract_feature/cub200_acil_r5_feat'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path,'{}.pkl'.format(id)),'wb') as f:
                pickle.dump(data_point,f,-1)
            print(id)
            id+=1
        # exit()
            # self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



@TRAINER_REGISTRY.register()
class ZeroshotCLIP_CIFAR_100(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.classnames = classnames
        # task_split = {0:10,1:20,2:30,3:40,4:50,5:60,6:70,7:80,8:90,9:100}
        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
    
        # with open("noval_classes_better.json") as f:
        #     gpt_data = json.load(f)
  

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load("ViT-B/16", device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]

        all_answer = {}
        # for i in range(len(templates)):
        #     temp = templates[i]
        #     prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        #     print(f"Prompts: {prompts}")
        #     prompts = torch.cat([clip.tokenize(p) for p in prompts])
        #     prompts = prompts.to(self.device)

        #     with torch.no_grad():
        #         text_features = clip_model.encode_text(prompts)
        #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     text_features = text_features.cpu()
        #     all_answer[i] = text_features
        # with open("text_encoding_hard.pkl",'wb') as f:
        #     pickle.dump(all_answer,f,-1)
        # exit()


        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        # sentences_feat = {}

        # for key in gpt_data:
        #     class_sentences = gpt_data[key]
        #     prompts = torch.cat([clip.tokenize(p) for p in class_sentences])
        #     prompts = prompts.to(self.device)
        #     with torch.no_grad():
        #         text_features = clip_model.encode_text(prompts)
        #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     sentences_feat[key] = text_features.cpu().numpy()
        # with open("novel_class_gpt_sentence_feats.pkl","wb") as f:
        #     pickle.dump(sentences_feat,f,-1)
        # exit()


        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)r
        cfg = self.cfg
        classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.classnames = classnames
        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
        self.lab2cname = {x:self.classnames[x] for x in range(100)}
        return 
        # task_id_now = self.cfg.TRAINER.TASK_ID
        # tfm_train = build_transform(self.cfg, is_train=True)
        # cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        # tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        # train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        # test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        # self.train_loader_x = train_loader
        # # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = test_loader  # optional, can be None
        # self.test_loader = test_loader

        # self.num_classes = 100
        # self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # # self.num_source_domains = dm.num_source_domains
          # dict {label: classname}

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


    def model_inference_feats(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        return image_features



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        # tfm_test = build_transform(self.cfg, is_train=False)
        tfm_test = build_transform(self.cfg, is_train=True)
        # cifar100 = CIFAR100_FEW(tfm_test)
        cifar_100 =  CIFAR100_FEW_CL(shot=5,tfm=tfm_test,task_id=self.cfg.TRAINER.TASK_ID,mode='train')
        
        data_loader = torch.utils.data.DataLoader(cifar_100,batch_size=1,num_workers=4)

        # if split is None:
        #     split = self.cfg.TEST.SPLIT

        # if split == "val" and self.val_loader is not None:
        #     data_loader = self.val_loader
        # else:
        #     split = "test"  # in case val_loader is None
        #     data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        ans = 0
        id = 0
        # for j in range(4):
        ans_list =[ ]
        for idx, input in enumerate(tqdm(data_loader)):

            
            # input, label = self.parse_batch_test(batch)
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
            output = self.model_inference_feats(image)

            image_features = output.cpu()

            label = label.cpu()

            data_point = {'feats':image_features.numpy(),'label':label.numpy()}
            # print(data_point)
            # ans_list.append(data_point)
            save_path = 'extract_feature/cifar100_task{}_feat_full'.format(self.cfg.TRAINER.TASK_ID)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path,'{}.pkl'.format(id)),'wb') as f:
                pickle.dump(data_point,f,-1)
            # print(id)
            id+=1
        exit()
            # self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



class SlimageNet_FEW(TorchDataset):

    def __init__(self, shots,select_class_list,tfm) -> None:

        self.tfm = tfm
        image_data_root = 'SlimageNet64/test'

        images_list = []
        labels_list = []

        for i in range(5):
            class_id = select_class_list[i]
            image_path = os.path.join(image_data_root,class_id)
            image_candi = os.listdir(image_path)

            idx_list = random.sample(image_candi,shots)

            # images_list.extend(idx_list)

            for j in range(len(idx_list)):
                images_list.append(os.path.join(image_data_root,class_id,idx_list[j]))

            for j in range(len(idx_list)):
                labels_list.append(i)

    

        self.images_list = images_list
        self.labeled_list = labels_list

        # print( self.images_list)
        # print()


        self.len = len(images_list)



    

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
class ZeroshotCLIP_Sl(TrainerX):
    def build_model(self):
        cfg = self.cfg

        data_root = 'SlimageNet64/test'
        class_list = os.listdir(data_root)
        class_list = random.sample(class_list,5)


        classnames = []

        class_nid = []
        
        with open('SlimageNet64/label_to_imagenet_class_mappings.txt','r') as f:
            image_split = f.readlines()
            for i in range(len(image_split)):
                image_split[i] = image_split[i].replace('\n','')
                ori_id,class_id,class_name = image_split[i].split(" ")
                class_id = 'n'+class_id.zfill(4)
                if class_id in class_list:
                    classnames.append(class_name)
                    class_nid.append(class_id)

            

        # classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        self.classnames = classnames
        self.class_ids = class_nid
        # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        # self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        # classnames = classnames[:self.task_id_end]
    


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load("ViT-B/16", device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        classnames = [['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)r
        cfg = self.cfg
        classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        self.classnames = classnames
        task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
        self.lab2cname = {x:self.classnames[x] for x in range(200)}
        return 
        # task_id_now = self.cfg.TRAINER.TASK_ID
        # tfm_train = build_transform(self.cfg, is_train=True)
        # cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        # tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        # train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        # test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        # self.train_loader_x = train_loader
        # # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = test_loader  # optional, can be None
        # self.test_loader = test_loader

        # self.num_classes = 100
        # self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # # self.num_source_domains = dm.num_source_domains
          # dict {label: classname}

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        tfm_test = build_transform(self.cfg, is_train=False)
  
        sum_acc = 0
        for i in range(600):


            cfg = self.cfg

            data_root = 'SlimageNet64/test'
            class_list = os.listdir(data_root)
            class_list = random.sample(class_list,5)


            classnames = []

            class_nid = []
            
            with open('SlimageNet64/label_to_imagenet_class_mappings.txt','r') as f:
                image_split = f.readlines()
                for i in range(len(image_split)):
                    image_split[i] = image_split[i].replace('\n','')
                    ori_id,class_id,class_name = image_split[i].split(" ")
                    class_id = 'n'+class_id.zfill(4)
                    if class_id in class_list:
                        classnames.append(class_name)
                        class_nid.append(class_id)

                

            # classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
            self.classnames = classnames
            self.class_ids = class_nid
            # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
            # self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
            # classnames = classnames[:self.task_id_end]
        


            print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            clip_model = load_clip_to_cpu(cfg)
            clip_model.to(self.device)

            _,self.preprocess = clip.load("ViT-B/16", device=self.device)

            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            print(f"Prompts: {prompts}")
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to(self.device)

            with torch.no_grad():
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self.text_features = text_features
            self.clip_model = clip_model
        # cifar100 = CIFAR100_FEW(tfm_test)
            cub_200 =  SlimageNet_FEW(shots=5,select_class_list=self.class_ids,tfm=tfm_test)

            print(len(cub_200))
            
            data_loader = torch.utils.data.DataLoader(cub_200,batch_size=36,num_workers=4)

            # if split is None:
            #     split = self.cfg.TEST.SPLIT

            # if split == "val" and self.val_loader is not None:
            #     data_loader = self.val_loader
            # else:
            #     split = "test"  # in case val_loader is None
            #     data_loader = self.test_loader

            print(f"Evaluate on the *{split}* set")
            ans = 0
            for idx, input in enumerate(tqdm(data_loader)):

                
                # input, label = self.parse_batch_test(batch)
                image = input[0].to(self.device)
                label = input[1].to(self.device)
        
                output = self.model_inference(image)
    
    
                
                self.evaluator.process(output, label)
        

            results = self.evaluator.evaluate()
            acc_one = results['accuracy']
            sum_acc+=acc_one
            # print(sum_acc/(i+1))


        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]




class AWA2(TorchDataset):

    def __init__(self, tfm, set_name='known') -> None:

        self.tfm = tfm
        image_data_root = 'AWA2/JPEGImages'
        if set=='known':
            test_class_file = "AWA2/trainclasses.txt"
        else:
            test_class_file = "AWA2/test.txt"

        with open(test_class_file,'r') as f:
            class_list = f.readlines()
            for i in range(len(class_list)):
                class_list[i] = class_list[i].replace('\n','')
        images_list = []
        labels_list = []

        for i in range(len(class_list)):
            class_name = class_list[i]
            image_list_path = os.path.join(image_data_root,class_name)
            image_candi = os.listdir(image_list_path)

            # images_list.extend(idx_list)

            for j in range(len(image_candi)):
                images_list.append(os.path.join(image_data_root,class_name,image_candi[j]))

                labels_list.append(i)

        self.images_list = images_list
        self.labeled_list = labels_list
        self.len = len(images_list)


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
class ZeroshotCLIP_AWA2(TrainerX):
    def build_model(self):
        cfg = self.cfg



        classnames = ["chimpanzee","giant_panda","leopard","persian_cat","pig","hippopotamus","humpback_whale","raccoon","rat","seal"]

        # classnames = ['antelope', 'grizzly_bear', 'killer_whale', 'beaver', 'dalmatian', 'horse', 'german_shepherd', 'blue_whale', 'siamese_cat', 'skunk', 'mole', 'tiger', 'moose', 'spider_monkey', 'elephant', 
        #               'gorilla', 'ox', 'fox', 'sheep', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'weasel', 'otter', 'buffalo', 'zebra', 'deer', 'bobcat', 'lion', 'mouse', 'polar_bear', 
        #               'collie', 'walrus', 'cow', 'dolphin']
        # class_nid = []


            

        # classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        self.classnames = classnames
        # self.class_ids = class_nid
    


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)r
        cfg = self.cfg
        classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        self.classnames = classnames
        task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
        self.lab2cname = {x:self.classnames[x] for x in range(200)}
        return 
        # task_id_now = self.cfg.TRAINER.TASK_ID
        # tfm_train = build_transform(self.cfg, is_train=True)
        # cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        # tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        # train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        # test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        # self.train_loader_x = train_loader
        # # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = test_loader  # optional, can be None
        # self.test_loader = test_loader

        # self.num_classes = 100
        # self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # # self.num_source_domains = dm.num_source_domains
          # dict {label: classname}

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()
        tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100 = CIFAR100_FEW(tfm_test)
        cub_200 =  AWA2(tfm=tfm_test, set_name='known')
        
        data_loader = torch.utils.data.DataLoader(cub_200,batch_size=100,num_workers=4)


        print(f"Evaluate on the *{split}* set")
        ans = 0
        for idx, input in enumerate(tqdm(data_loader)):

            
            # input, label = self.parse_batch_test(batch)
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
            output = self.model_inference(image)
  
 
            
            self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    



class SUN397_FEW_cl(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode,class_per_task=10,test_model='all',test_task=0) -> None:

        self.tfm = tfm

        root = 'SUN397'

        image_root = os.path.join(root,'images')

        print(image_root)

        class_name_file = os.path.join(root,'split/ClassName.txt')
        train_list_file = os.path.join(root,'split/Training_01.txt')
        test_list_file = os.path.join(root,'split/Testing_01.txt')
        with open(class_name_file,'r') as f:
            class_name_list = f.readlines()
            for i in range(len(class_name_list)):
                class_name_list[i] = class_name_list[i].replace('\n','')
                # image_id,is_train = image_split[i].split(" ")
                # image_id_split[image_id] = eval(is_train)
        self.class_name_to_id = {}
        for i in range(len(class_name_list)):
            self.class_name_to_id[class_name_list[i]] = i

        # print(class_name_list)

        
            
        self.mode = mode
        self.all_class_num = 397
        self.base_class_num=197

        train_test_split = 'CUB_200_2011/train_test_split.txt'
        self.class_per_task = class_per_task
        self.novel_task_len = int((self.all_class_num-self.base_class_num)/class_per_task)
        self.task_len = self.novel_task_len+1
        

        



        self.images_list = []
        self.labeled_list = []

        class_to_image_dict = {x:[] for x in range(397)}

        if mode=='train':

            task_split = [[] for x in range(self.task_len)]

            for i in range(self.base_class_num):
                task_split[0].append(i)

            for i in range(1,self.task_len):
                for j in range(self.class_per_task):
                    task_split[i].append(self.base_class_num+(i-1)*self.class_per_task+j)

            select_class_id = task_split[task_id]

            print(select_class_id)
            self.end_class_id = select_class_id[-1]
           
            # self.class_idx_dict = {x:[] for x in select_class_id}


            
            
            with open(train_list_file,'r') as f:
                train_list_all = f.readlines()
                for i in range(len(train_list_all)):
                    train_list_all[i] = train_list_all[i].replace('\n','')
                    class_list_split = train_list_all[i].split("/")[:-1]
                    class_name = "/".join(class_list_split)
                    class_id = self.class_name_to_id[class_name]
                    class_to_image_dict[class_id].append(train_list_all[i])
        

            # print(select_class_id)
            # for key in image_id_path_dict:
            #     if image_id_split[key]==1:
            #         label = image_id_label_dict[key]
            #         if label in  self.class_idx_dict:
            #             self.class_idx_dict[label].append(key)


            for c in select_class_id:
                image_path_list = class_to_image_dict[c]
                # print(idx_list)
                # print(shot)
                if c>=self.base_class_num:
                    
                    image_path_list = random.sample(image_path_list,shot)
                for image_path in image_path_list:
                    image_path_full = os.path.join(image_root,image_path[1:])
                  
                    self.images_list.append(image_path_full)
                    self.labeled_list.append(c)
            # print(self.images_list)
            # exit()
  

            self.shot = shot

            self.len = len(self.images_list)
            # self.mean_feat_list = {}
            # for class_id in range(self.end_class_id-9):
            #     GD_all = self.GD[class_id]
            #     mean_feat = []
            #     for dim in range(len(GD_all)):
            #         dim_param = GD_all[dim]
            #         mean = dim_param['mean']
            #         # std = dim_param['std']
            #         mean_feat.append(mean)
            #     self.mean_feat_list[class_id] = np.array(mean_feat)

        else:
            if test_model=='all':
                self.data = []
                task_to_id_end = {0:self.base_class_num}
                start = self.base_class_num+self.class_per_task
                for i in range(1,self.task_len):
                    task_to_id_end[i]=start
                    start+=self.class_per_task

                select_class_id=[x for x in range(task_to_id_end[task_id])]

                with open(test_list_file,'r') as f:
                    test_list_all = f.readlines()
                    for i in range(len(test_list_all)):
                        test_list_all[i] = test_list_all[i].replace('\n','')
                        class_list_split = test_list_all[i].split("/")[:-1]
                        class_name = "/".join(class_list_split)
                        class_id = self.class_name_to_id[class_name]
                        class_to_image_dict[class_id].append(test_list_all[i])


                print(select_class_id)

                for c in select_class_id:
                    image_path_list = class_to_image_dict[c]
                    # print(idx_list)
                    # print(shot)
                    # if c>=self.base_class_num:
                        
                        # image_path_list = random.sample(image_path_list,shot)
                    for image_path in image_path_list:
                        image_path_full = os.path.join(image_root,image_path[1:])
                        self.images_list.append(image_path_full)
                        self.labeled_list.append(c)

                self.shot = shot

                self.len = len(self.images_list)
            else:
                task_split = [[] for x in range(11)]

                for i in range(100):
                    task_split[0].append(i)

                for i in range(1,11):
                    for j in range(10):
                        task_split[i].append(100+(i-1)*10+j)

                select_class_id = task_split[test_task]

                print(select_class_id)

                for key in image_id_path_dict:
                    if image_id_split[key]==0:
                        label = image_id_label_dict[key]
                        if label in  select_class_id:
                            self.images_list.append(image_id_path_dict[key])
                            self.labeled_list.append(label)

                self.shot = shot

                self.len = len(self.images_list)


    

    def __getitem__(self,idx):

        img_name = self.images_list[idx]
        label = self.labeled_list[idx]
        # img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.tfm:
            image = self.tfm(image)

        # if self.mode=='train':
        #     pseudo_label = []
        #     pseudo_feat_list = []
        #     have_select = 0
        #     while True:
           
        #         select_class = random.randint(0,self.end_class_id-self.class_per_task)

        #             # seed = random.uniform(0,1)
        #             # if seed>0.5:
                
        #         GD_all = self.GD[select_class]
        #         pseudo_feat = []
        #         for dim in range(len(GD_all)):
        #             dim_param = GD_all[dim]
        #             # mean = self.GD_full[select_class][dim]['mean']
        #             mean = dim_param['mean']
        #             # std = max(dim_param['std'],self.GD_text[select_class][dim]['std'])
        #             # std = self.GD_full[select_class][dim]['std']
        #             std = dim_param['std']
        #             pseudo_value = np.random.normal(mean, std, 1)[0]
        #             pseudo_feat.append(pseudo_value)
        #         pseudo_feat = np.array(pseudo_feat)

        #         pseudo_feat = pseudo_feat / np.linalg.norm(pseudo_feat)

        #         pseudo_feat_list.append(pseudo_feat)
        #         pseudo_label.append(select_class)
      
        #         have_select+=1
        #         if have_select==7:
        #             break
        #     pseudo_label = np.array(pseudo_label)
        #     pseudo_feat_list = np.array(pseudo_feat_list)
            
        #     return image,label,pseudo_feat_list,pseudo_label
        return image,label



    def __len__(self):
        return self.len





@TRAINER_REGISTRY.register()
class ZeroshotCLIP_SUN397(TrainerX):
    def build_model(self):
        cfg = self.cfg



        

        # classnames = ['antelope', 'grizzly_bear', 'killer_whale', 'beaver', 'dalmatian', 'horse', 'german_shepherd', 'blue_whale', 'siamese_cat', 'skunk', 'mole', 'tiger', 'moose', 'spider_monkey', 'elephant', 
        #               'gorilla', 'ox', 'fox', 'sheep', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'weasel', 'otter', 'buffalo', 'zebra', 'deer', 'bobcat', 'lion', 'mouse', 'polar_bear', 
        #               'collie', 'walrus', 'cow', 'dolphin']
        # class_nid = []


            

        # classnames = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']
        classnames = self.classnames[:self.class_end]
        # self.class_ids = class_nid
    


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)r
        cfg = self.cfg
        classnames = [
    'abbey',
    'airplane cabin',
    'airport terminal',
    'alley',
    'amphitheater',
    'amusement arcade',
    'amusement park',
    'anechoic chamber',
    'apartment building outdoor',
    'apse indoor',
    'aquarium',
    'aqueduct',
    'arch',
    'archive',
    'arrival gate outdoor',
    'art gallery',
    'art school',
    'art studio',
    'assembly line',
    'athletic field outdoor',
    'atrium public',
    'attic',
    'auditorium',
    'auto factory',
    'badlands',
    'badminton court indoor',
    'baggage claim',
    'bakery shop',
    'balcony exterior',
    'balcony interior',
    'ball pit',
    'ballroom',
    'bamboo forest',
    'banquet hall',
    'bar',
    'barn',
    'barndoor',
    'baseball field',
    'basement',
    'basilica',
    'basketball court outdoor',
    'bathroom',
    'batters box',
    'bayou',
    'bazaar indoor',
    'bazaar outdoor',
    'beach',
    'beauty salon',
    'bedroom',
    'berth',
    'biology laboratory',
    'bistro indoor',
    'boardwalk',
    'boat deck',
    'boathouse',
    'bookstore',
    'booth indoor',
    'botanical garden',
    'bow window indoor',
    'bow window outdoor',
    'bowling alley',
    'boxing ring',
    'brewery indoor',
    'bridge',
    'building facade',
    'bullring',
    'burial chamber',
    'bus interior',
    'butchers shop',
    'butte',
    'cabin outdoor',
    'cafeteria',
    'campsite',
    'campus',
    'canal natural',
    'canal urban',
    'candy store',
    'canyon',
    'car interior backseat',
    'car interior frontseat',
    'carrousel',
    'casino indoor',
    'castle',
    'catacomb',
    'cathedral indoor',
    'cathedral outdoor',
    'cavern indoor',
    'cemetery',
    'chalet',
    'cheese factory',
    'chemistry lab',
    'chicken coop indoor',
    'chicken coop outdoor',
    'childs room',
    'church indoor',
    'church outdoor',
    'classroom',
    'clean room',
    'cliff',
    'cloister indoor',
    'closet',
    'clothing store',
    'coast',
    'cockpit',
    'coffee shop',
    'computer room',
    'conference center',
    'conference room',
    'construction site',
    'control room',
    'control tower outdoor',
    'corn field',
    'corral',
    'corridor',
    'cottage garden',
    'courthouse',
    'courtroom',
    'courtyard',
    'covered bridge exterior',
    'creek',
    'crevasse',
    'crosswalk',
    'cubicle office',
    'dam',
    'delicatessen',
    'dentists office',
    'desert sand',
    'desert vegetation',
    'diner indoor',
    'diner outdoor',
    'dinette home',
    'dinette vehicle',
    'dining car',
    'dining room',
    'discotheque',
    'dock',
    'doorway outdoor',
    'dorm room',
    'driveway',
    'driving range outdoor',
    'drugstore',
    'electrical substation',
    'elevator door',
    'elevator interior',
    'elevator shaft',
    'engine room',
    'escalator indoor',
    'excavation',
    'factory indoor',
    'fairway',
    'fastfood restaurant',
    'field cultivated',
    'field wild',
    'fire escape',
    'fire station',
    'firing range indoor',
    'fishpond',
    'florist shop indoor',
    'food court',
    'forest broadleaf',
    'forest needleleaf',
    'forest path',
    'forest road',
    'formal garden',
    'fountain',
    'galley',
    'game room',
    'garage indoor',
    'garbage dump',
    'gas station',
    'gazebo exterior',
    'general store indoor',
    'general store outdoor',
    'gift shop',
    'golf course',
    'greenhouse indoor',
    'greenhouse outdoor',
    'gymnasium indoor',
    'hangar indoor',
    'hangar outdoor',
    'harbor',
    'hayfield',
    'heliport',
    'herb garden',
    'highway',
    'hill',
    'home office',
    'hospital',
    'hospital room',
    'hot spring',
    'hot tub outdoor',
    'hotel outdoor',
    'hotel room',
    'house',
    'hunting lodge outdoor',
    'ice cream parlor',
    'ice floe',
    'ice shelf',
    'ice skating rink indoor',
    'ice skating rink outdoor',
    'iceberg',
    'igloo',
    'industrial area',
    'inn outdoor',
    'islet',
    'jacuzzi indoor',
    'jail cell',
    'jail indoor',
    'jewelry shop',
    'kasbah',
    'kennel indoor',
    'kennel outdoor',
    'kindergarden classroom',
    'kitchen',
    'kitchenette',
    'labyrinth outdoor',
    'lake natural',
    'landfill',
    'landing deck',
    'laundromat',
    'lecture room',
    'library indoor',
    'library outdoor',
    'lido deck outdoor',
    'lift bridge',
    'lighthouse',
    'limousine interior',
    'living room',
    'lobby',
    'lock chamber',
    'locker room',
    'mansion',
    'manufactured home',
    'market indoor',
    'market outdoor',
    'marsh',
    'martial arts gym',
    'mausoleum',
    'medina',
    'moat water',
    'monastery outdoor',
    'mosque indoor',
    'mosque outdoor',
    'motel',
    'mountain',
    'mountain snowy',
    'movie theater indoor',
    'museum indoor',
    'music store',
    'music studio',
    'nuclear power plant outdoor',
    'nursery',
    'oast house',
    'observatory outdoor',
    'ocean',
    'office',
    'office building',
    'oil refinery outdoor',
    'oilrig',
    'operating room',
    'orchard',
    'outhouse outdoor',
    'pagoda',
    'palace',
    'pantry',
    'park',
    'parking garage indoor',
    'parking garage outdoor',
    'parking lot',
    'parlor',
    'pasture',
    'patio',
    'pavilion',
    'pharmacy',
    'phone booth',
    'physics laboratory',
    'picnic area',
    'pilothouse indoor',
    'planetarium outdoor',
    'playground',
    'playroom',
    'plaza',
    'podium indoor',
    'podium outdoor',
    'pond',
    'poolroom establishment',
    'poolroom home',
    'power plant outdoor',
    'promenade deck',
    'pub indoor',
    'pulpit',
    'putting green',
    'racecourse',
    'raceway',
    'raft',
    'railroad track',
    'rainforest',
    'reception',
    'recreation room',
    'residential neighborhood',
    'restaurant',
    'restaurant kitchen',
    'restaurant patio',
    'rice paddy',
    'riding arena',
    'river',
    'rock arch',
    'rope bridge',
    'ruin',
    'runway',
    'sandbar',
    'sandbox',
    'sauna',
    'schoolhouse',
    'sea cliff',
    'server room',
    'shed',
    'shoe shop',
    'shopfront',
    'shopping mall indoor',
    'shower',
    'skatepark',
    'ski lodge',
    'ski resort',
    'ski slope',
    'sky',
    'skyscraper',
    'slum',
    'snowfield',
    'squash court',
    'stable',
    'stadium baseball',
    'stadium football',
    'stage indoor',
    'staircase',
    'street',
    'subway interior',
    'subway station platform',
    'supermarket',
    'sushi bar',
    'swamp',
    'swimming pool indoor',
    'swimming pool outdoor',
    'synagogue indoor',
    'synagogue outdoor',
    'television studio',
    'temple east asia',
    'temple south asia',
    'tennis court indoor',
    'tennis court outdoor',
    'tent outdoor',
    'theater indoor procenium',
    'theater indoor seats',
    'thriftshop',
    'throne room',
    'ticket booth',
    'toll plaza',
    'topiary garden',
    'tower',
    'toyshop',
    'track outdoor',
    'train railway',
    'train station platform',
    'tree farm',
    'tree house',
    'trench',
    'underwater coral reef',
    'utility room',
    'valley',
    'van interior',
    'vegetable garden',
    'veranda',
    'veterinarians office',
    'viaduct',
    'videostore',
    'village',
    'vineyard',
    'volcano',
    'volleyball court indoor',
    'volleyball court outdoor',
    'waiting room',
    'warehouse indoor',
    'water tower',
    'waterfall block',
    'waterfall fan',
    'waterfall plunge',
    'watering hole',
    'wave',
    'wet bar',
    'wheat field',
    'wind farm',
    'windmill',
    'wine cellar barrel storage',
    'wine cellar bottle storage',
    'wrestling ring indoor',
    'yard',
    'youth hostel',
]

        task_id_now = self.cfg.TRAINER.TASK_ID
        class_per_task = 10
        base_class = 197
        all_class = 397
        task_num = int((all_class-base_class)/class_per_task)+1
        self.task_end_id= {0:base_class}
        start = base_class+class_per_task
        for i in range(1,task_num):
            self.task_end_id[i]=start
            start+=class_per_task

        self.class_end = self.task_end_id[task_id_now]

        self.classnames = classnames
        # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        # self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        # classnames = classnames[:self.task_id_end]
        self.lab2cname = {x:self.classnames[x] for x in range(397)}
        return 
        # task_id_now = self.cfg.TRAINER.TASK_ID
        # tfm_train = build_transform(self.cfg, is_train=True)
        # cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        # tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        # train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        # test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        # self.train_loader_x = train_loader
        # # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = test_loader  # optional, can be None
        # self.test_loader = test_loader

        # self.num_classes = 100
        # self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # # self.num_source_domains = dm.num_source_domains
          # dict {label: classname}

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits



    def model_inference_feats(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        return image_features

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()
        tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100 = CIFAR100_FEW(tfm_test)
        sun_397 =  SUN397_FEW_cl(5,tfm_test,task_id=self.cfg.TRAINER.TASK_ID,mode='test',class_per_task=10,test_model='all',test_task=0)
        
        data_loader = torch.utils.data.DataLoader(sun_397,batch_size=1,num_workers=4)


        print(f"Evaluate on the *{split}* set")
        ans = 0
        id = 0
        # for idx, input in enumerate(tqdm(data_loader)):

            
        #     # input, label = self.parse_batch_test(batch)
        #     image = input[0].to(self.device)
        #     label = input[1].to(self.device)
    
        #     # output = self.model_inference(image)


        #     output = self.model_inference_feats(image)

        #     image_features = output.cpu()

        #     label = label.cpu()

        #     data_point = {'feats':image_features.numpy(),'label':label.numpy()}
        #     # ans_list.append(data_point)
        #     save_path = 'extract_feature/sun397_task{}_feat'.format(self.cfg.TRAINER.TASK_ID)
        #     if not os.path.exists(save_path):
        #         os.mkdir(save_path)
        #     with open(os.path.join(save_path,'{}.pkl'.format(id)),'wb') as f:
        #         pickle.dump(data_point,f,-1)
        #     print(id)
        #     id+=1
  
 
        # exit()


        for idx, input in enumerate(tqdm(data_loader)):

            
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
            output = self.model_inference(image)
  
 
            
            self.evaluator.process(output, label)


  
 
            
            # self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]



@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model







class MiniImageNet(TorchDataset):

    def __init__(self, tfm,task_id,mode,test_model='all'):


        self.IMAGE_PATH = 'miniimagenet/images'
        self.SPLIT_PATH = 'miniimagenet/split'
        self.tfm = tfm
        
        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}

        csv_path = osp.join(self.SPLIT_PATH, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]


        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        

        # if mode=='train':
    

    
        #     self.data, self.targets = self.SelectfromClasses(self.data, self.targets, np.arange(task_split[0]))

        #     # self.data = []
        #     # self.targets = []
        #     for i in range(1,task_id+1):
        #         index_path = 'index_list/mini_imagenet/session_{}.txt'.format(i+1)
        #         tmp_data, tmp_data_target = self.SelectfromTxt(self.data2label, index_path)
        #         # for j in range(20):
        #         self.data.extend(tmp_data)
        #         self.targets.extend(tmp_data_target)
  
        # else:

        #     self.data, self.targets = self.SelectfromClasses(self.data, self.targets, np.arange(task_split[task_id]))


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
        
            ind_cl = np.where(i == targets)[0]

            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.tfm(Image.open(path).convert('RGB'))
        return image, targets




@TRAINER_REGISTRY.register()
class ZeroshotCLIP_miniImageNet(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = ['house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode', 'king_crab', 'dugong', 'Walker_hound', 'Ibizan_hound', 'Saluki', 'golden_retriever', 'Gordon_setter', 'komondor', 'boxer', 'Tibetan_mastiff', 'French_bulldog', 'malamute', 'dalmatian', 'Newfoundland', 'miniature_poodle', 'white_wolf', 'African_hunting_dog', 'Arctic_fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros_beetle', 'ant', 'black-footed_ferret', 'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle', 'bookshop', 'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'clog', 'cocktail_shaker', 'combination_lock', 'crate', 'cuirass', 'dishrag', 'dome', 'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 'hair_slide', 'holster', 'horizontal_bar', 'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile', 'mixing_bowl', 'oboe', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'poncho', 'prayer_rug', 'reel', 'school_bus', 'scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web', 'stage', 'tank', 'theater_curtain', 'tile_roof', 'tobacco_shop', 'unicycle', 'upright', 'vase', 'wok', 'worm_fence', 'yawl', 'street_sign', 'consomme', 'trifle', 'hotdog', 'orange', 'cliff', 'coral_reef', 'bolete', 'ear']
        self.classnames = classnames
        # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        # class_per_task=10
        # task_num = int(200/class_per_task)
        # self.task_end_id= {0:10}
        # start = 10+class_per_task
        # for i in range(1,task_num):
        #     self.task_end_id[i]=start
        #     start+=class_per_task
        self.task_end_id = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        # task_split = {}
        # for i in range(20):
        #     task_split[i] = (i+1)*10
        self.task_id_end = self.task_end_id[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:self.task_id_end]
    
        # with open("noval_classes_better.json") as f:
        #     gpt_data = json.load(f)
  

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        _,self.preprocess = clip.load("ViT-B/16", device=self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]



        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        # sentences_feat = {}

        # for key in gpt_data:
        #     class_sentences = gpt_data[key]
        #     prompts = torch.cat([clip.tokenize(p) for p in class_sentences])
        #     prompts = prompts.to(self.device)
        #     with torch.no_grad():
        #         text_features = clip_model.encode_text(prompts)
        #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     sentences_feat[key] = text_features.cpu().numpy()
        # with open("novel_class_gpt_sentence_feats.pkl","wb") as f:
        #     pickle.dump(sentences_feat,f,-1)
        # exit()


        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)r
        cfg = self.cfg
        classnames = ['house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode', 'king_crab', 'dugong', 'Walker_hound', 'Ibizan_hound', 'Saluki', 'golden_retriever', 'Gordon_setter', 'komondor', 'boxer', 'Tibetan_mastiff', 'French_bulldog', 'malamute', 'dalmatian', 'Newfoundland', 'miniature_poodle', 'white_wolf', 'African_hunting_dog', 'Arctic_fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros_beetle', 'ant', 'black-footed_ferret', 'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle', 'bookshop', 'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'clog', 'cocktail_shaker', 'combination_lock', 'crate', 'cuirass', 'dishrag', 'dome', 'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 'hair_slide', 'holster', 'horizontal_bar', 'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile', 'mixing_bowl', 'oboe', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'poncho', 'prayer_rug', 'reel', 'school_bus', 'scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web', 'stage', 'tank', 'theater_curtain', 'tile_roof', 'tobacco_shop', 'unicycle', 'upright', 'vase', 'wok', 'worm_fence', 'yawl', 'street_sign', 'consomme', 'trifle', 'hotdog', 'orange', 'cliff', 'coral_reef', 'bolete', 'ear']
        self.classnames = classnames
        # task_split = {0:100,1:110,2:120,3:130,4:140,5:150,6:160,7:170,8:180,9:190,10:200}
        # task_split = {}
        # for i in range(20):
        #     task_split[i] = (i+1)*10
        # self.task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        self.task_end_id = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}

        self.task_id_end = self.task_end_id[self.cfg.TRAINER.TASK_ID]

        classnames = classnames[:self.task_id_end]
        self.lab2cname = {x:self.classnames[x] for x in range(100)}
        return 
        # task_id_now = self.cfg.TRAINER.TASK_ID
        # tfm_train = build_transform(self.cfg, is_train=True)
        # cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        # tfm_test = build_transform(self.cfg, is_train=False)
        # cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        # train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        # test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        # self.train_loader_x = train_loader
        # # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = test_loader  # optional, can be None
        # self.test_loader = test_loader

        # self.num_classes = 100
        # self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # # self.num_source_domains = dm.num_source_domains
          # dict {label: classname}

    def model_inference(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


    def model_inference_feats(self, image):


   
        image_features = self.clip_model.encode_image(image)
  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        return image_features



    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        # tfm_test = build_transform(self.cfg, is_train=False)
        tfm_test = build_transform(self.cfg, is_train=True)
        # cifar100 = CIFAR100_FEW(tfm_test)
        cub_200 =  MiniImageNet(tfm=tfm_test,task_id=self.cfg.TRAINER.TASK_ID,mode='train')
        
        data_loader = torch.utils.data.DataLoader(cub_200,batch_size=1,num_workers=4)

        # if split is None:
        #     split = self.cfg.TEST.SPLIT

        # if split == "val" and self.val_loader is not None:
        #     data_loader = self.val_loader
        # else:
        #     split = "test"  # in case val_loader is None
        #     data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        ans = 0
        id = 0
        # for j in range(4):
        ans_list =[ ]
        for idx, input in enumerate(tqdm(data_loader)):
            image = input[0].to(self.device)
            label = input[1].to(self.device)
    
            # output = self.model_inference(image)
  
 
            
            # self.evaluator.process(output, label)

            
            # input, label = self.parse_batch_test(batch)
        #     image = input[0].to(self.device)
        #     label = input[1].to(self.device)
    
            output = self.model_inference_feats(image)

            image_features = output.cpu()

            label = label.cpu()

            data_point = {'feats':image_features.numpy(),'label':label.numpy()}

            save_path = 'extract_feature/miniImageNet_full'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path,'{}.pkl'.format(id)),'wb') as f:
                pickle.dump(data_point,f,-1)
            print(id)
            id+=1
        # exit()
            # self.evaluator.process(output, label)
       

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]