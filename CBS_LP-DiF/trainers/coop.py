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
from tqdm import tqdm
import random
import os
from PIL import Image
import pickle
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        if cfg.TRAINER.COOP.LOAD_CTX:
            with open(cfg.TRAINER.COOP.LOAD_CTX_PATH,'rb') as f:
                ctx = pickle.load(f)
            self.ctx = nn.Parameter(ctx)
        else:
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]


        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # print(embedding)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
    


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


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

    def forward(self, image,pseudo_feat=None):


        image_features = self.image_encoder(image.type(self.dtype))
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


class CustomCLIP_feat(nn.Module):
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

    def forward(self, image):

        if image.shape[-1]==512:
            image_features = image.half()

        else:


            image_features = self.image_encoder(image.type(self.dtype))
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


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.clip_model = clip_model
        self.model = CustomCLIP(cfg, classnames, clip_model)

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
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
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



class CIFAR100_FEW(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode) -> None:

        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        task_id_end = task_split[task_id]


        if mode=='train':
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=True, train=True,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in range(task_id_end):
                idx_list = self.class_idx_dict[c]
                # if c>=60:
                sample_few_idx = random.sample(idx_list,shot)
                for id in sample_few_idx:
                    self.data.append(self.cifar100[id])
                # else:
                #     for id in idx_list:
                #         self.data.append(self.cifar100[id])


            self.shot = shot

            
        else:
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=True, train=False,transform=tfm)
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



class CIFAR100_full(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode) -> None:

        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        task_id_end = task_split[task_id]


        if mode=='train':
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=True, train=True,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in range(task_id_end):

                idx_list = self.class_idx_dict[c]
                if c>=60:
                    sample_few_idx = []
                    tmp = random.sample(idx_list,shot)
                    for j in range(20):
                        sample_few_idx.extend(tmp)
                    idx_list = sample_few_idx
                for id in idx_list:
                    self.data.append(id)
                # else:
                #     for id in idx_list:
                #         self.data.append(self.cifar100[id])


            self.shot = shot

            
        else:
            self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=True, train=False,transform=tfm)
            self.class_idx_dict = {x:[] for x in range(100)}
            for i in range(len(self.cifar100)):
                image,label = self.cifar100[i]
                self.class_idx_dict[label].append(i)

            self.data = []

            for c in range(task_id_end):
                idx_list = self.class_idx_dict[c]
                for id in idx_list:
                    self.data.append(id)

        self.len = len(self.data)

    

    def __getitem__(self,index):
        return self.cifar100[self.data[index]]


    def __len__(self):
        return self.len



class CIFAR100_FEW_cl(TorchDataset):

    def __init__(self, shot,tfm,task_id,mode,class_per_task,test_model='all',test_task=0) -> None:

        # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)

        # with open("cifar100_train.pkl",'wb') as f:
        #     pickle.dump(cifar100,f,-1)
        # exit()

        self.class_per_task = class_per_task
        self.novel_task_len = int(40/class_per_task)
        self.task_len = self.novel_task_len+1
        self.mode = mode
        self.task_id = task_id

        if mode=='train':

            with open('cifar100_GD.pkl','rb') as f:
                self.GD = pickle.load(f)
            task_split = [[] for x in range(9)]

            for i in range(60):
                task_split[0].append(i)

            for i in range(1,9):
                for j in range(5):
                    task_split[i].append(60+(i-1)*5+j)
            # self.class_idx_dict = 
            select_class_id = task_split[task_id]
            # print(select_class_id)
            cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=True,transform=tfm)
            self.class_idx_dict = {x:[] for x in select_class_id}
            self.end_class_id = select_class_id[-1]

            print(select_class_id)
            for i in range(len(cifar100)):
                image,label = cifar100[i]

                if label in self.class_idx_dict:

                    self.class_idx_dict[label].append(i)

            self.data = []
            # print(select_class_id)
            for c in select_class_id:
                idx_list = self.class_idx_dict[c]
                if c>=60:
                    idx_list = random.sample(idx_list,shot)
                for id in idx_list:
                    self.data.append(cifar100[id])

            self.shot = shot

            self.len = len(self.data)
        else:
            if test_model=='all':
                self.data = []
                task_to_id_end = {0:60}
                start = 65
                for i in range(1,9):
                    task_to_id_end[i]=start
                    start+=5

                select_class_id=[x for x in range(task_to_id_end[task_id])]

                self.end_class_id = select_class_id[-1]


                print(select_class_id)




                cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"),download=False, train=False,transform=tfm)

                self.class_idx_dict = {x:[] for x in select_class_id}
                for i in range(len(cifar100)):
                    image,label = cifar100[i]

                    if label in select_class_id:

                        self.data.append(cifar100[i])
                self.len = len(self.data)
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


    

    def __getitem__(self,index):
        if self.mode=='train' and self.task_id>0:
            pseudo_label = []
            pseudo_feat_list = []
            have_select = 0
            while True:
           
                select_class = random.randint(0,self.end_class_id-self.class_per_task)
                # print(select_class)

                    # seed = random.uniform(0,1)
                    # if seed>0.5:
                
                GD_all = self.GD[select_class]
                pseudo_feat = []
                for dim in range(len(GD_all)):
                    dim_param = GD_all[dim]
                    # mean = self.GD_full[select_class][dim]['mean']
                    mean = dim_param['mean']
                    # std = max(dim_param['std'],self.GD_text[select_class][dim]['std'])
                    # std = self.GD_full[select_class][dim]['std']
                    std = dim_param['std']
                    pseudo_value = np.random.normal(mean, std, 1)[0]
                    pseudo_feat.append(pseudo_value)
                pseudo_feat = np.array(pseudo_feat)

                pseudo_feat = pseudo_feat / np.linalg.norm(pseudo_feat)

                pseudo_feat_list.append(pseudo_feat)
                pseudo_label.append(select_class)
      
                have_select+=1
                if have_select==1:
                    break
            pseudo_label = np.array(pseudo_label)
            pseudo_feat_list = np.array(pseudo_feat_list)
            
            return self.data[index][0],self.data[index][1],pseudo_feat_list,pseudo_label
        return self.data[index]


    def __len__(self):
        return self.len







    

class DatasetWrapper_My(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


@TRAINER_REGISTRY.register()
class CoOp_CIFAR(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]


    # def after_epoch(self):
    #     last_epoch = (self.epoch + 1) == self.max_epoch
    #     do_test = not self.cfg.TEST.NO_TEST
    #     meet_checkpoint_freq = (
    #         (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
    #         if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
    #     )

    #     if do_test and self.cfg.TEST.FINAL_MODEL == "best_val" and :
    #         curr_result = self.test(split="val")
    #         is_best = curr_result > self.best_result
    #         if is_best:
    #             self.best_result = curr_result
    #             self.save_model(
    #                 self.epoch,
    #                 self.output_dir,
    #                 val_result=curr_result,
    #                 model_name="model-best.pth.tar"
    #             )

    #     if meet_checkpoint_freq or last_epoch:
    #         self.save_model(self.epoch, self.output_dir)


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)
        task_id_now = self.cfg.TRAINER.TASK_ID
        tfm_train = build_transform(self.cfg, is_train=True)
        cifar100_train = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_train,task_id=task_id_now,mode='train')


        tfm_test = build_transform(self.cfg, is_train=False)
        cifar100_test = CIFAR100_full(shot=self.cfg.DATASET.NUM_SHOTS,tfm=tfm_test,task_id=task_id_now,mode='test')


        train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=128,num_workers=4,drop_last=False)

        test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,num_workers=4,drop_last=False)

        self.train_loader_x = train_loader
        # self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = test_loader  # optional, can be None
        self.test_loader = test_loader

        self.num_classes = 100
        self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # self.num_source_domains = dm.num_source_domains
        self.lab2cname = {x:self.classnames[x] for x in range(100)}  # dict {label: classname}

        # self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        task_split = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        task_id_end = task_split[self.cfg.TRAINER.TASK_ID]
        classnames = classnames[:task_id_end]
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

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
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):

        
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
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
         
    
            output = self.model_inference(image)
  
 
            
            self.evaluator.process(output, label)
       

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



@TRAINER_REGISTRY.register()
class CoOp_CIFAR_CL(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]


    # def after_epoch(self):
    #     last_epoch = (self.epoch + 1) == self.max_epoch
    #     do_test = not self.cfg.TEST.NO_TEST
    #     meet_checkpoint_freq = (
    #         (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
    #         if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
    #     )

    #     if do_test and self.cfg.TEST.FINAL_MODEL == "best_val" and :
    #         curr_result = self.test(split="val")
    #         is_best = curr_result > self.best_result
    #         if is_best:
    #             self.best_result = curr_result
    #             self.save_model(
    #                 self.epoch,
    #                 self.output_dir,
    #                 val_result=curr_result,
    #                 model_name="model-best.pth.tar"
    #             )

    #     if meet_checkpoint_freq or last_epoch:
    #         self.save_model(self.epoch, self.output_dir)

    # def train(self):
    #     self.start_epoch = 1
    #     self.before_train()
    #     # for i in range(9):
    #     #     self.start_epoch = 1
    #     #     if i==0:
    #     #         # print(self.model.module.device)
    #     #         continue
    #     #     if i>=1:
             


    #             # print(self.model.module.device)
    #             # self.model.module.prompt_learner.update_prompt(self.cfg,current_class,self.clip_model)
         
    #     for self.epoch in range(self.start_epoch, self.max_epoch):
    #         if self.epoch>1:
    #             self.model.module.prompt_learner.class_end=65

    #             self.model.module.tokenized_prompts = self.model.module.prompt_learner.tokenized_prompts[:65]
    #             train_set_task = CIFAR100_FEW_cl(shot=self.cfg.DATASET.NUM_SHOTS,tfm=self.tfm_train,task_id = 1,mode='train')
         
    #             test_set_task = CIFAR100_FEW_cl(shot=self.cfg.DATASET.NUM_SHOTS,tfm=self.tfm_test,task_id = 1,mode='test')

    #             train_loader = torch.utils.data.DataLoader(train_set_task,batch_size=32,num_workers=4,drop_last=False)
    #             test_loader = torch.utils.data.DataLoader(test_set_task,batch_size=100,num_workers=4,drop_last=False)
    #             self.train_loader_x = train_loader
    #             self.test_loader = test_loader

    #             # current_class = self.classnames_all[:self.task_end_id[1]]
    #             # self.model.module.prompt_learner.update_prompt(self.cfg,current_class,self.clip_model)
    #         self.before_epoch()
    #         self.run_epoch()
    #         self.after_epoch()
        
        
    #     self.after_train()           
               

            # super().train()


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        # dm = DataManager(self.cfg,dataset_wrapper=DatasetWrapper_My)
        self.tfm_train = build_transform(self.cfg, is_train=True)
        task_id_now = self.cfg.TRAINER.TASK_ID
        print(task_id_now)
        train_set_task0 = CIFAR100_FEW_cl(shot=self.cfg.DATASET.NUM_SHOTS,tfm=self.tfm_train,task_id = task_id_now,mode='train',class_per_task=5)
        self.tfm_test = build_transform(self.cfg, is_train=False)
        test_set_task0 = CIFAR100_FEW_cl(shot=self.cfg.DATASET.NUM_SHOTS,tfm=self.tfm_test,task_id = task_id_now,mode='test',test_model='all',class_per_task=5)

        train_loader = torch.utils.data.DataLoader(train_set_task0,batch_size=50,num_workers=4,drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set_task0,batch_size=100,num_workers=4,drop_last=False)

  

        self.train_loader_x = train_loader
        # self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = test_loader  # optional, can be None
        self.test_loader = test_loader

        self.num_classes = 100
        self.classnames = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # self.num_source_domains = dm.num_source_domains
        self.lab2cname = {x:self.classnames[x] for x in range(100)}  # dict {label: classname}

        # self.dm = dm

    def build_model(self):
        cfg = self.cfg
        task_id_now = self.cfg.TRAINER.TASK_ID
        self.task_id = task_id_now
        self.task_end_id = {0:60,1:65,2:70,3:75,4:80,5:85,6:90,7:95,8:100}
        class_end = self.task_end_id[task_id_now]
        self.classnames_all = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        class_task = self.classnames_all[:class_end]

        


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
            self.model = nn.DataParallel(self.model,device_ids=[0, 1, 2,3,4,5,6])

    def forward_backward(self, batch):

        if self.task_id==0:
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



            pseudo_feat = pseudo_feat.view(-1,512)
            
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
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
         
    
            output = self.model_inference(image)
  
 
            
            self.evaluator.process(output, label)
       

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











