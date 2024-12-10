# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import copy
import random
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data import DataLoader
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
from torch.utils.data import Dataset as TorchDataset
import logging
from PIL import Image
import pickle
import copy

from .utils import row_norms, kmeans_plus_plus_opt

al_dict = {}



def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Calculate the KL divergence between two multivariate Gaussian distributions.

    Parameters:
    mu_p (np.ndarray): Mean of the first Gaussian distribution.
    sigma_p (np.ndarray): Standard deviation of the first Gaussian distribution.
    mu_q (np.ndarray): Mean of the second Gaussian distribution.
    sigma_q (np.ndarray): Standard deviation of the second Gaussian distribution.

    Returns:
    float: KL divergence between the two distributions.
    """
    # Ensure the inputs are numpy arrays
    mu_p, sigma_p, mu_q, sigma_q = map(np.array, [mu_p, sigma_p, mu_q, sigma_q])

    # Calculate KL divergence
    divergence = np.sum(np.log(sigma_q / sigma_p) + 
                        (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5)

    return divergence





def calculate_GD(data_list,begin_class,class_num):
    GD_dict = {x:[] for x in range(begin_class,begin_class+class_num)}
    feat_dict = {x:[] for x in range(begin_class,begin_class+class_num)}
    dim=512


    for i in range(len(data_list)):
        data = data_list[i]
    
        feat = data['feats']
    
        label = data['label']
        # print(label)

        feat_dict[label].append(feat)


    for key in feat_dict:
        feat_list = np.array(feat_dict[key])
        if feat_list.shape[0]<2:
            continue

        for j in range(dim):
            f_j = feat_list[:,j]
            mean = np.mean(f_j)
            std = np.std(f_j,ddof=1)
            GD_dict[key].append({'mean':mean,'std':std})

    return GD_dict



def calculate_GD_one_class(feat_list):

    feat_list = np.array(feat_list)


    # if feat_list.shape[0]<2:
    #     continue

    dim =512

    GD_dict = []

    for j in range(dim):
        f_j = feat_list[:,j]
        mean = np.mean(f_j)
        std = np.std(f_j,ddof=1)
        if std==0:
            std=1e-6
        GD_dict.append({'mean':mean,'std':std})

    return GD_dict



def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls

    return decorator


def get_strategy(sample, *args):
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)

def read_file_to_list(file_name):

    with open(file_name, 'r') as file:
        lines = [line.strip() for line in file]

    return lines


class QueryDataset(TorchDataset):

    def __init__(self, image_path_list,label_list,tfm,mode='query',task_id=None,save_dir=None, CL_method='None') -> None:


        self.tfm = tfm
        self.CL_method = CL_method
        self.mode = mode
        self.task_id = task_id


        if mode=='train' and CL_method=='joint':
            pre_root = os.path.join(save_dir,'session_{}.txt')
            # pre_root = 'CUB_200_2011/ACIL_entropy/R_5/SEED0/
            old_images_list = []
            old_label_list = []
            # self.shot = shot
            for i in range(task_id):
                lines = read_file_to_list(pre_root.format(i))

                for line in lines:
                    image_path, image_label = line.split(" ")
                    old_images_list.append(image_path)
                    old_label_list.append(int(image_label))
            image_path_list.extend(old_images_list)
            label_list.extend(old_label_list)
        elif mode=='train' and CL_method=='LP-DiF':
            GD_path = os.path.join(save_dir,'GD.pkl')
            
            max_key = -1
            if self.task_id > 0:
                with open(GD_path,'rb') as f:
                    self.GD = pickle.load(f)

                for key in self.GD:
                    max_key = max(key,max_key)
                self.old_class_end = max_key


        assert len(image_path_list) == len(label_list), 'image number must match label number'

        

        self.images_list = image_path_list
        self.labeled_list = label_list

        self.len = len(self.images_list)

    

    def __getitem__(self,idx):

        img_name = self.images_list[idx]
        label = self.labeled_list[idx]
        # img_name = os.path.join(self.root_dir,img_name)
        image = Image.open(img_name).convert('RGB')

        if self.tfm:
            image = self.tfm(image)

        if self.mode=='train' and self.CL_method =="LP-DiF" and self.task_id>0:
            pseudo_label = []
            pseudo_feat_list = []
            have_select = 0
            while True:
           
                select_class = random.randint(0,self.old_class_end)
                if select_class not in self.GD:
                    continue
                if len(self.GD[select_class])==0:
                    continue

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
                if have_select==7:
                    break
            pseudo_label = np.array(pseudo_label)
            pseudo_feat_list = np.array(pseudo_feat_list)

            # print(pseudo_feat_list.shape)
            
            return image,label,pseudo_feat_list,pseudo_label


        return image,label



    def __len__(self):
        return self.len




class QueryDataset_CIFAR(TorchDataset):

    def __init__(self, idx_list, all_data, mode='query',task_id=None,save_dir=None, CL_method='None') -> None:


        # self.tfm = tfm
        self.CL_method = CL_method
        self.mode = mode
        self.task_id = task_id

        self.data = []


        if mode=='train' and CL_method=='joint':
            pre_root = os.path.join(save_dir,'session_{}.txt')
            # pre_root = 'CUB_200_2011/ACIL_entropy/R_5/SEED0/
            old_images_list = []
            old_label_list = []
            # self.shot = shot
            for i in range(task_id):
                lines = read_file_to_list(pre_root.format(i))

                for line in lines:
                    image_path, image_label = line.split(" ")
                    old_images_list.append(image_path)
                    old_label_list.append(int(image_label))
            image_path_list.extend(old_images_list)
            label_list.extend(old_label_list)
        elif mode=='train' and CL_method=='LP-DiF':
            GD_path = os.path.join(save_dir,'GD.pkl')
            
            max_key = -1
            if self.task_id > 0:
                with open(GD_path,'rb') as f:
                    self.GD = pickle.load(f)

                for key in self.GD:
                    max_key = max(key,max_key)
                self.old_class_end = max_key


        # assert len(image_path_list) == len(label_list), 'image number must match label number'

        
        for idx in idx_list:
            self.data.append(all_data[idx])

        # print(self.data)
        # self.images_list = image_path_list
        # self.labeled_list = label_list

        self.len = len(self.data)

    

    def __getitem__(self,idx):



        if self.mode=='train' and self.task_id>0 and self.CL_method =="LP-DiF":
            pseudo_label = []
            pseudo_feat_list = []
            have_select = 0
            while True:
           
                select_class = random.randint(0,self.old_class_end)
                if select_class not in self.GD:
                    continue
                if len(self.GD[select_class])==0:
                    continue

        
                
                GD_all = self.GD[select_class]
                pseudo_feat = []
                for dim in range(len(GD_all)):
                    dim_param = GD_all[dim]

                    mean = dim_param['mean']

                    std = dim_param['std']
                    pseudo_value = np.random.normal(mean, std, 1)[0]
                    pseudo_feat.append(pseudo_value)
                pseudo_feat = np.array(pseudo_feat)

                pseudo_feat = pseudo_feat / np.linalg.norm(pseudo_feat)

                pseudo_feat_list.append(pseudo_feat)
                pseudo_label.append(select_class)
      
                have_select+=1
                if have_select==7:
                    break
            pseudo_label = np.array(pseudo_label)
            pseudo_feat_list = np.array(pseudo_feat_list)
            
            return self.data[idx][0],self.data[idx][1],pseudo_feat_list,pseudo_label
        return self.data[idx]



    def __len__(self):
        return self.len






class SamplingStrategy:
    """
    Sampling Strategy wrapper class
    """

    def __init__(self, dset, model, device, cfg, train_process, eval_process, save_dir=None, CL_method='None'):
        self.dset = dset
        self.num_classes = dset.class_per_task
        self.model = copy.deepcopy(model) # initialized with source model
        self.device = device
        self.cfg = cfg
        self.train_process = train_process

        self.eval_process = eval_process
        self.task_id = dset.task_id
        self.save_dir = save_dir
        self.CL_method = CL_method
    
    @property
    def idxs_lb(self):
        return self.dset.idxs_lb

    def update(self, idxs_active):
        self.dset.idxs_lb[idxs_active] = True

    def query(self, n, epoch):
        pass

    def pred(self, idxs=None, with_emb=False, ignore_old=False):
        if idxs is None:
            idxs = np.arange(self.dset.len)[~self.idxs_lb]

        data_loader = self.build_unlabel_loaders(idxs)
        self.model.eval()
        all_log_probs = []
        all_scores = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                data, target = batch[0].to(self.device), batch[1].to(self.device)
                if with_emb:
                   scores, embs = self.model(image=data, with_emb=True)
                   all_embs.append(embs.cpu())
                else:
                   scores = self.model(image=data, with_emb=False)
                if ignore_old and self.dset.task_id>0:
                    class_per_task = self.dset.class_per_task
                    scores = scores[:,-class_per_task:]
                log_probs = nn.LogSoftmax(dim=1)(scores)
                all_log_probs.append(log_probs)
                all_scores.append(scores)

        all_log_probs = torch.cat(all_log_probs)
        all_probs = torch.exp(all_log_probs)
        all_scores = torch.cat(all_scores)
        if with_emb:
            all_embs = torch.cat(all_embs)
            return idxs, all_probs, all_log_probs, all_scores, all_embs
        else:
            return idxs, all_probs, all_log_probs, all_scores

    def build_unlabel_loaders(self, idxs):
        if isinstance(idxs, np.ndarray):
            idxs = idxs.tolist()
        if self.dset.name=='cifar':
            # label_idx = [ind for ind in np.where(self.dset.idxs_lb)[0].tolist()]
       
            query_set = QueryDataset_CIFAR(idxs, self.dset.data, mode='query',task_id=self.task_id,save_dir=self.save_dir, CL_method=self.CL_method)            
        else:
            image_path_list = [self.dset.images_list[ind] for ind in idxs]
            label_list = [self.dset.labeled_list[ind] for ind in idxs]

            query_set = QueryDataset(image_path_list, label_list, self.eval_process,mode='query')

        loader = DataLoader(query_set,batch_size=1,num_workers=1)

        return loader

    def build_label_loaders(self):
        if self.dset.name=='cifar':
            label_idx = [ind for ind in np.where(self.dset.idxs_lb)[0].tolist()]
       
            train_set = QueryDataset_CIFAR(label_idx, self.dset.data, mode='train',task_id=self.task_id,save_dir=self.save_dir, CL_method=self.CL_method)
        else:
            image_path_list = [self.dset.images_list[ind] for ind in np.where(self.dset.idxs_lb)[0].tolist()]
            label_list = [self.dset.labeled_list[ind] for ind in np.where(self.dset.idxs_lb)[0].tolist()]
            train_set = QueryDataset(image_path_list, label_list, self.train_process,mode='train',task_id=self.task_id,save_dir=self.save_dir, CL_method=self.CL_method)
    
        
        loader = DataLoader(train_set, batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE, num_workers=1, drop_last=True, shuffle=True)

        return loader




@register_strategy('fully')
class FullySampling(SamplingStrategy):
    """
    Uniform sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process,save_dir, CL_method):
        super(FullySampling, self).__init__(dset, model, device, cfg, train_process,eval_process,save_dir,CL_method)

    def query(self, n, epoch=None):
        return np.where(self.idxs_lb == 0)[0]
    

@register_strategy('random')
class RandomSampling(SamplingStrategy):
    """
    Uniform sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process,save_dir, CL_method):
        super(RandomSampling, self).__init__(dset, model, device, cfg, train_process,eval_process,save_dir,CL_method)

    def query(self, n, epoch=None):
        return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)


@register_strategy('random_force_balance')
class RandomForceBalanceSampling(SamplingStrategy):
    """
    Uniform sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process,save_dir, CL_method):
        super(RandomForceBalanceSampling, self).__init__(dset, model, device, cfg, train_process,eval_process,save_dir,CL_method)
        self.unlabeled_pool_size = self.dset.len
        self.class_per_task = self.dset.class_per_task
        self.mean_per_class = self.unlabeled_pool_size//self.class_per_task
        self.select_average_per_class = cfg.AL.ROUND
        self.select_all = self.class_per_task * self.select_average_per_class

    def query(self, n):
        if self.dset.name=='cifar':
            labeled_list = [self.dset.data[i][1] for i in range(len(self.dset.data))]
            # print(labeled_list)
            classes_set = list(set(labeled_list))
    
        else:
            classes_set = list(set(self.dset.labeled_list))
            labeled_list = self.dset.labeled_list
        unlabeled_idxs = np.where(self.idxs_lb == 0)[0]
        sample_all = []
        for class_id in classes_set:
            candidate = []
            for idx in unlabeled_idxs:
                if labeled_list[idx] == class_id:
                    candidate.append(idx)
            candidate = np.array(candidate)
            select_idx = np.random.choice(candidate, self.select_average_per_class, replace=False)
            sample_all.extend(select_idx)
        print(len(sample_all))

        return np.array(sample_all)


@register_strategy('entropy')
class EntropySampling(SamplingStrategy):
    """
    Implements entropy based sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(EntropySampling, self).__init__(dset, model, device, cfg, train_process, eval_process, save_dir, CL_method)

    def query(self, n, epoch):
        idxs_unlabeled, all_probs, all_log_probs, _ = self.pred()
        # Compute entropy
        entropy = -(all_probs * all_log_probs).sum(1)
        q_idxs = (entropy).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]



@register_strategy('entropy_force_balance')
class EntropyForceBalanceSampling(SamplingStrategy):
    """
    Implements entropy based sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(EntropyForceBalanceSampling, self).__init__(dset, model, device, cfg, train_process, eval_process, save_dir, CL_method)


    def query(self, n, epoch):
        idxs_unlabeled, all_probs, all_log_probs, _ = self.pred(ignore_old=True)
 
        
        # Compute entropy
        entropy = -(all_probs * all_log_probs).sum(1)
 
        
        q_idxs = (entropy).sort(descending=True)[1]
        
        q_idxs = q_idxs.cpu().numpy()
        idxs_unlabeled_sort = idxs_unlabeled[q_idxs]

        classes_set = list(set(self.dset.labeled_list))

        visit_dict = {classes_set[i]:0 for i in range(len(classes_set))}

        select_idxs = []

        for idx in idxs_unlabeled_sort:
            labeled = self.dset.labeled_list[idx]
            if visit_dict[labeled]==0:
                visit_dict[labeled] = 1
                select_idxs.append(idx)
        return np.array(select_idxs)



@register_strategy('margin')
class MarginSampling(SamplingStrategy):
    """
    Implements margin based sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(MarginSampling, self).__init__(dset, model, device, cfg, train_process,eval_process, save_dir,CL_method)

    def query(self, n, epoch):
        idxs_unlabeled, all_probs, _, _ = self.pred()
        # Compute BvSB margin
        top2 = torch.topk(all_probs, 2).values
        BvSB_scores = 1-(top2[:,0] - top2[:,1]) # use minus for descending sorting
        q_idxs = (BvSB_scores).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]


@register_strategy('margin_force_balance')
class MarginForceBalanceSampling(SamplingStrategy):
    """
    Implements margin based sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(MarginForceBalanceSampling, self).__init__(dset, model, device, cfg, train_process,eval_process, save_dir,CL_method)

    def query(self, n, epoch):
        idxs_unlabeled, all_probs, _, _ = self.pred(ignore_old=True)
        # Compute BvSB margin
        top2 = torch.topk(all_probs, 2).values
        BvSB_scores = 1-(top2[:,0] - top2[:,1]) # use minus for descending sorting
        q_idxs = (BvSB_scores).sort(descending=True)[1]
        q_idxs = q_idxs.cpu().numpy()

        idxs_unlabeled_sort = idxs_unlabeled[q_idxs]

        classes_set = list(set(self.dset.labeled_list))

        visit_dict = {classes_set[i]:0 for i in range(len(classes_set))}

        select_idxs = []

        for i in range(idxs_unlabeled_sort.shape[0]):
            idx = idxs_unlabeled_sort[i]
            labeled = self.dset.labeled_list[idx]
            if visit_dict[labeled]==0:
                visit_dict[labeled] = 1
                select_idxs.append(idx)

        print(select_idxs)

        return np.array(select_idxs)



@register_strategy('leastConfidence')
class LeastConfidenceSampling(SamplingStrategy):
    def __init__(self, dset, model, device, cfg):
        super(LeastConfidenceSampling, self).__init__(dset, model, device, cfg)

    def query(self, n, epoch):
        idxs_unlabeled, all_probs, _, _ = self.pred()
        confidences = -all_probs.max(1)[0] # use minus for descending sorting
        q_idxs = (confidences).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]


@register_strategy('coreset')
class CoreSetSampling(SamplingStrategy):
    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(CoreSetSampling, self).__init__(dset, model, device, cfg, train_process,eval_process, save_dir,CL_method)

    def furthest_first(self, X, X_lb, n):
        m = np.shape(X)[0]
        if np.shape(X_lb)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_lb)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n, epoch):
        if self.dset.name!='cifar':
            idxs = np.arange(len(self.dset.images_list))
        else:
            idxs = np.arange(len(self.dset.data))
        idxs_unlabeled, _, _, _, all_embs = self.pred(idxs=idxs, with_emb=True)
        all_embs = all_embs.numpy()
        q_idxs = self.furthest_first(all_embs[~self.idxs_lb, :], all_embs[self.idxs_lb, :], n)
        return idxs_unlabeled[q_idxs]


@register_strategy('AADA')
class AADASampling(SamplingStrategy):
    """
    Implements Active Adversarial Domain Adaptation (https://arxiv.org/abs/1904.07848)
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(AADASampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        """
        s(x) = frac{1-G*_d}{G_f(x))}{G*_d(G_f(x))} [Diversity] * H(G_y(G_f(x))) [Uncertainty]
        """
        self.model.eval()
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=4, batch_size=64,
                                                  drop_last=False)

        # Get diversity and entropy
        all_log_probs, all_scores = [], []
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores = self.model(data)
                log_probs = nn.LogSoftmax(dim=1)(scores)
                all_scores.append(scores)
                all_log_probs.append(log_probs)

        all_scores = torch.cat(all_scores)
        all_log_probs = torch.cat(all_log_probs)

        all_probs = torch.exp(all_log_probs)
        disc_scores = nn.Softmax(dim=1)(self.discrim(all_scores))
        # Compute diversity
        self.D = torch.div(disc_scores[:, 0], disc_scores[:, 1])
        # Compute entropy
        self.E = -(all_probs * all_log_probs).sum(1)
        scores = (self.D * self.E).sort(descending=True)[1]
        # Sample from top-2 % instances, as recommended by authors
        top_N = max(int(len(scores) * 0.02), n)
        q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)

        return idxs_unlabeled[q_idxs]


@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
    """
    Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(BADGESampling, self).__init__(dset, model, device, cfg, train_process,eval_process, save_dir,CL_method)

    def query(self, n, epoch):

        if self.dset.name!='cifar':
            unlabeled_pool_size = len(self.dset.images_list)
        else:
            unlabeled_pool_size = len(self.dset.data)
        idxs_unlabeled = np.arange(unlabeled_pool_size)[~self.idxs_lb]
        data_loader = self.build_unlabel_loaders(idxs_unlabeled)
        self.model.eval()

        emb_dim = 512

        tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes*(self.task_id+1)])  # 这个其实应该是logits...
        tgt_pen_emb = torch.zeros([len(data_loader.sampler), emb_dim])
        tgt_lab = torch.zeros(len(data_loader.sampler))
        tgt_preds = torch.zeros(len(data_loader.sampler))
        batch_sz = 1

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                data, target = batch[0].to(self.device), batch[1].to(self.device)
                e1, e2 = self.model(image=data, with_emb=True)
                tgt_pen_emb[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
                tgt_emb[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
                tgt_lab[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e1.shape[0])] = target
                tgt_preds[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1,
                                                         keepdim=True).squeeze()

        # Compute uncertainty gradient
        tgt_scores = nn.Softmax(dim=1)(tgt_emb)
        tgt_scores_delta = torch.zeros_like(tgt_scores)
        tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1

        # Uncertainty embedding
        badge_uncertainty = (tgt_scores - tgt_scores_delta)

        # Seed with maximum uncertainty example
        max_norm = row_norms(badge_uncertainty.cpu().numpy()).argmax()

        _, q_idxs = kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n,
                                               init=[max_norm])

        return idxs_unlabeled[q_idxs]


@register_strategy('kmeans')
class KmeansSampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    def __init__(self, dset, model, device, cfg):
        super(KmeansSampling, self).__init__(dset, model, device, cfg)

    def query(self, n, epoch):
        idxs_unlabeled, _, _, _, all_embs = self.pred(with_emb=True)
        all_embs = all_embs.numpy()

        # Run weighted K-means over embeddings
        km = KMeans(n_clusters=n)
        km.fit(all_embs)

        # use below code to match CLUE implementation
        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, all_embs)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        return idxs_unlabeled[q_idxs]


@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(CLUESampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)
        self.random_state = np.random.RandomState(1234)
        self.T = 0.1

    def query(self, n, epoch):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS, \
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()

        if 'LeNet' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 500
        elif 'ResNet34' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 512
        elif 'ResNet50' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 256

        # Get embedding of target instances
        tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = get_embedding(self.model, data_loader, self.device,
                                                                       self.num_classes, \
                                                                       self.cfg, with_emb=True, emb_dim=emb_dim)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()
        tgt_scores = torch.softmax(tgt_emb / self.T, dim=-1)
        tgt_scores += 1e-8
        sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()

        # Run weighted K-means over embeddings
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        return idxs_unlabeled[q_idxs]


@register_strategy('ProbCover')
class ProbCoverSampling(SamplingStrategy):
    """
    Implements ProbCover
    """

    def __init__(self, dset, model, device,  cfg, train_process,eval_process, save_dir,CL_method):
        super(ProbCoverSampling, self).__init__(dset, model, device, cfg, train_process,eval_process, save_dir,CL_method)
        self.delta = -1
        self.ratio = 0.5
    
    def construct_graph(self, all_embs, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        import pandas as pd
        
        xs, ys, ds = [], [], []
        batch_size = len(all_embs)
        # distance computations are done in GPU
        cuda_feats = torch.tensor(all_embs).cuda()
        for i in range(len(all_embs) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            print(dist.max().item(), dist.min().item(), dist.mean().item()) 
            if self.delta == -1:
                self.delta = self.ratio * dist.mean().item()
                logging.info(f'Start constructing graph using delta={self.delta}')
            
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        logging.info(f'Finished constructing graph using delta={self.delta}')
        logging.info(f'Graph contains {len(df)} edges.')
        return df
    
    def query(self, n, epoch):
        # idxs = np.arange(len(self.dset.train_idx))
        # idxs = np.arange(len(self.dset.images_list))

        if self.dset.name!='cifar':
            unlabeled_pool_size = len(self.dset.images_list)
        else:
            unlabeled_pool_size = len(self.dset.data)

        idxs = np.arange(unlabeled_pool_size)
        idxs_unlabeled, _, _, _, all_embs = self.pred(idxs=idxs, with_emb=True)
        all_embs = all_embs.numpy()
        graph_df = self.construct_graph(all_embs)
        
        selected = []
        pre_select = np.where(self.idxs_lb)[0]
        print("Pre Select:", pre_select)
        edge_from_seen = np.isin(graph_df.x, pre_select)
        covered_samples = graph_df.y[edge_from_seen].unique()
        cur_df = graph_df[(~np.isin(graph_df.y, covered_samples))]
        all_sample_len = unlabeled_pool_size
        for i in range(n):
            coverage = len(covered_samples) / all_sample_len
            degrees = np.bincount(cur_df.x, minlength=all_sample_len)
            cur = degrees.argmax()
            logging.info(f'Iteration is {i}, select {cur}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            print(f'Iteration is {i}, select {cur}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]
            
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)
        
        logging.info("Finish selecting.")
        print("Finish selecting.")
        print("Select:", selected)
        return selected




def get_nn(features, num_neighbors):
    import faiss
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_




@register_strategy('Typiclust')
class TypiclustSampling(SamplingStrategy):
    """
    Implements Typiclust. Adopt from https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/typiclust.py
    """
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(TypiclustSampling, self).__init__(dset, model, device, cfg, train_process,eval_process, save_dir,CL_method)
    
    def init_features_and_clusters(self, features, budget, num_labeled):
        print(f"budget: {budget}, num_labeled: {num_labeled}")
        num_clusters = min(num_labeled + budget, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clustering.')
        clusters = kmeans(features, num_clusters=num_clusters)
        print(f'Finished clustering into {num_clusters} clusters.')
        return clusters
    
    def query(self, n, epoch):
        import pandas as pd

        if self.dset.name!='cifar':
            unlabeled_pool_size = len(self.dset.images_list)
        else:
            unlabeled_pool_size = len(self.dset.data)
        idxs = np.arange(unlabeled_pool_size)
        # idxs = np.arange(len(self.dset.train_x))  # 获取所有的embedding, 如果embedding不改变，可以考虑缓存它以加快速度。
        _, _, _, _, all_embs = self.pred(idxs=idxs, with_emb=True)
        all_embs = all_embs.numpy()
        labels = self.init_features_and_clusters(all_embs, n, sum(self.idxs_lb))
        existing_indices = np.where(self.idxs_lb)  # 这里是所有已标注样本的index
        
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []
        for i in range(n):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id  # 从最大的cluster依次往后选，直到选完或者再次从头循环
            indices = (labels == cluster).nonzero()[0]
            rel_feats = all_embs[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            assert self.idxs_lb[idx] != 1, f"Sample {idx} has been selected."
            selected.append(idx)
            labels[idx] = -1
        
        logging.info("Finish selecting.")
        print("Finish selecting.")
        print("Select:", selected)
        return selected




@register_strategy('PCB')
class PCBSampling(SamplingStrategy):
    """
    Pseudo-Class Balance sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process):
        super(PCBSampling, self).__init__(dset, model, device, cfg, train_process,eval_process)
        self.all_class_num = dset.all_class_num
        self.class_per_task = dset.class_per_task

    def query(self, n, p_ulb):
        _, _, _, scores = self.pred(idxs=p_ulb, with_emb=False)
        scores = F.softmax(scores, dim=-1)
        scores[:,:-self.class_per_task] = 0
        scores = scores.argmax(dim=-1).detach().cpu().numpy().tolist()

        print(scores)

        samples_by_cls= {}
        for idx, p_label in enumerate(scores):
            if p_label in samples_by_cls:
                samples_by_cls[p_label].append(p_ulb[idx])
            else:
                samples_by_cls[p_label] = [p_ulb[idx]]
        
        # 从每类中随机抽取
        print(f"We found {len(samples_by_cls)} pseudo class when doing PCB.")
        # 维护一个现有标注集的结果
        current_dist = {}
        # data_source = [self.dset.train_x[ind] for ind in np.where(self.dset.idxs_lb)[0].tolist()]
        image_path_list = [self.dset.images_list[ind] for ind in np.where(self.dset.idxs_lb)[0].tolist()]
        label_list = [self.dset.labeled_list[ind] for ind in np.where(self.dset.idxs_lb)[0].tolist()]
        for label in label_list:
            # 伪标签集里有的才统计，不然没必要统计
            if label not in samples_by_cls: continue

            if label in current_dist:  
                current_dist[label] += 1
            else:
                current_dist[label] = 1
        current_dist = sorted(current_dist.items(), key=lambda x: x[1])
        current_dist = [list(it) for it in current_dist]

        active_set = []
        for idx in range(n):
            smallest_cls = current_dist[0][0]

            random.shuffle(samples_by_cls[smallest_cls])
            active_set.append(samples_by_cls[smallest_cls][0])
            samples_by_cls[smallest_cls] = samples_by_cls[smallest_cls][1:]

            # 维护现有的集合
            if len(samples_by_cls[smallest_cls]) == 0:  # 取完了
                current_dist = current_dist[1:]
            else:
                current_dist[0][1] += 1
                current_dist = sorted(current_dist, key=lambda x: x[1])  # 重新排序
        
        return active_set






@register_strategy('distribution_kmeans_random_discard_greedy')
class DistributionKMeansRandomDiscardGreedySampling(SamplingStrategy):
    """
    Implements entropy based sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(DistributionKMeansRandomDiscardGreedySampling, self).__init__(dset, model, device, cfg, train_process, eval_process, save_dir, CL_method)
        # if self.dset.name=='cifar':
        #     self.unlabeled_pool_size = len(self.dset.data)
        # else:
        self.unlabeled_pool_size = self.dset.len
        self.class_per_task = self.dset.class_per_task
        self.mean_per_class = self.unlabeled_pool_size//self.class_per_task
        self.select_average_per_class = cfg.AL.ROUND
        self.select_all = self.class_per_task * self.select_average_per_class


    def query(self):
        # idxs = np.arange(len(self.dset.data))
        # print(idxs)
        idxs_unlabeled, all_probs, _, _, all_embs = self.pred(with_emb=True, ignore_old=True)

        # print(idxs_unlabeled)

        feats_label = []

        feats_dict = {}

        idxs_dict = {}

        featall_emb_numpy = all_embs.numpy()

        class_per_task = self.dset.class_per_task

        # model = GaussianMixture(n_components=class_per_task)
        model = KMeans(n_clusters=class_per_task)

        model.fit(featall_emb_numpy)

        yhat = model.predict(featall_emb_numpy)

        for i in range(idxs_unlabeled.shape[0]):
            idx = idxs_unlabeled[i]

            feature = all_embs[i].numpy()

            label = yhat[i]

            data_point = {'feats':feature, 'label':label}
            feats_label.append(data_point)

            if label not in feats_dict:
                feats_dict[label] = []
                idxs_dict[label] = []
            
            feats_dict[label].append(feature)
            idxs_dict[label].append(idx)


        # class_per_task = self.dset.class_per_task
        begin_class = self.task_id * class_per_task

        GD_dict = calculate_GD(feats_label, 0, class_per_task)

        select_idx_all = []
        for key in feats_dict:
            fully_GD = GD_dict[key]

            if len(fully_GD)==0:
                select_idx_all.extend(idxs_dict[key])
                continue
                

            fully_GD_mean = []
            fully_GD_var = []
            for dim in range(len(fully_GD)):
                dim_point = fully_GD[dim]
                fully_GD_mean.append(dim_point['mean'])
                fully_GD_var.append(dim_point['std'])
            fully_GD_mean = torch.Tensor(fully_GD_mean)

            fully_GD_var = torch.Tensor(fully_GD_var)
            minn_kl = 1000000
            best_choice = None
            feats_per_classes = np.array(feats_dict[key])
            idxs_per_classes = np.array(idxs_dict[key])
            sample_num = math.ceil(self.select_average_per_class * len(feats_per_classes)/self.mean_per_class)

            if sample_num>1:

                # initialize
                best_sim = -100000
                best_choice = 0
                best_feat = None

                select_idx_per_class = []
                select_feats_per_class = []
                for j in range(len(feats_per_classes)):
                    select_feats_per_classes = torch.from_numpy(feats_per_classes[j])

                    select_idxs_per_classes = idxs_per_classes[j]

                    sim = torch.dot(select_feats_per_classes.float(), fully_GD_mean)

                    if sim>best_sim:
                        best_sim = sim
                        best_choice = copy.copy(select_idxs_per_classes)
                        best_feat = copy.copy(select_feats_per_classes.numpy())
                select_idx_per_class.append(best_choice)
                select_feats_per_class.append(best_feat)

                for k in range(sample_num-1):
                    minn_kl = 1000000
                    best_feat = None
                    best_choice = 0
                    for j in range(len(feats_per_classes)):
                        tmp_idx_per_classes = idxs_per_classes[j]
                        
                        if tmp_idx_per_classes in select_idx_per_class:
                            continue

                        tmp_feat_per_classes = feats_per_classes[j]
                        tmp_select_feats_per_class = copy.copy(select_feats_per_class)
                        tmp_select_feats_per_class.append(tmp_feat_per_classes)
                        sub_GD = calculate_GD_one_class(tmp_select_feats_per_class)
                        sub_GD_mean = []
                        sub_GD_var = []

                        for dim in range(len(sub_GD)):
                            dim_point = sub_GD[dim]
                            sub_GD_mean.append(dim_point['mean'])
                            sub_GD_var.append(dim_point['std'])
                        sub_GD_mean = torch.Tensor(sub_GD_mean)
                        sub_GD_var = torch.Tensor(sub_GD_var)

                        kl_divers = kl_divergence_gaussian(sub_GD_mean, sub_GD_var, fully_GD_mean, fully_GD_var)
                        if kl_divers<minn_kl:
                            minn_kl = kl_divers
                            best_choice = copy.copy(tmp_idx_per_classes)
                            best_feat = copy.copy(tmp_feat_per_classes)
                    select_idx_per_class.append(best_choice)
                    select_feats_per_class.append(best_feat)
                select_idx_all.extend(select_idx_per_class)


            else:
                best_sim = -100000
                best_choice = 0
                for j in range(len(feats_per_classes)):
                    select_feats_per_classes = torch.from_numpy(feats_per_classes[j])

                    select_idxs_per_classes = idxs_per_classes[j]

                    sim = torch.dot(select_feats_per_classes.float(), fully_GD_mean)

                    if sim>best_sim:
                        best_sim = sim
                        best_choice = copy.copy(select_idxs_per_classes)
                select_idx_all.append(best_choice)
        

                
        restrict = self.select_all

        if len(select_idx_all) > restrict:


            select_idx_all = random.sample(select_idx_all, restrict)
        

        return np.array(select_idx_all)






@register_strategy('dropquery')
class DropquerySampling(SamplingStrategy):
    """
    Implements entropy based sampling
    """

    def __init__(self, dset, model, device, cfg, train_process,eval_process, save_dir,CL_method):
        super(DropquerySampling, self).__init__(dset, model, device, cfg, train_process, eval_process, save_dir, CL_method)
        # if self.dset.name=='cifar':
        #     self.unlabeled_pool_size = len(self.dset.data)
        # else:
        self.unlabeled_pool_size = self.dset.len
        self.class_per_task = self.dset.class_per_task
        self.mean_per_class = self.unlabeled_pool_size//self.class_per_task


    def query(self, n, epoch):
        # idxs = np.arange(len(self.dset.data))
        # print(idxs)
        idxs_unlabeled, all_probs, _, _, all_embs = self.pred(with_emb=True, ignore_old=False)
        filter_feat = []
        filter_idxs = []
        all_preds = all_probs.argmax(dim=1)
        for i in range(all_embs.shape[0]):
            feat_one = all_embs[i].to(self.device)
            idx = idxs_unlabeled[i]
            pred = all_preds[i]
            # print("astar:",pred)
            num_zeroed = int(0.75 * feat_one.numel())

            incon = 0
            for time in range(3):
                feat_copy = copy.deepcopy(feat_one)
                indices = torch.randperm(feat_copy.numel())[:num_zeroed]
                
                feat_copy[indices] = 0.0
                feat_copy = feat_copy.unsqueeze(0).half()
                # print(feat_copy)
                tmp_scores = self.model.module.forward_pseudo(pseudo_feat=feat_copy)
                tmp_preds = nn.Softmax(dim=1)(tmp_scores).argmax(dim=1)
                # print("tmp preds:",tmp_preds)
                if pred!=tmp_preds[0]:
                    incon+=1
            if incon>2:
                filter_feat.append(all_embs[i].numpy())
                filter_idxs.append(idx)
            continue

        # print(idxs_unlabeled)

        feats_label = []

        feats_dict = {}

        idxs_dict = {}

        filter_feat = np.array(filter_feat)
        filter_idxs = np.array(filter_idxs)

        # featall_emb_numpy = all_embs.numpy()

        class_per_task = self.dset.class_per_task

        # model = GaussianMixture(n_components=class_per_task)
        model = KMeans(n_clusters=class_per_task)

        model.fit(filter_feat)

        yhat = model.predict(filter_feat)

        for i in range(filter_idxs.shape[0]):
            idx = filter_idxs[i]

            feature = filter_feat[i]

            label = yhat[i]
            # feats_label.append(data_point)

            if label not in feats_dict:
                feats_dict[label] = []
                idxs_dict[label] = []
            
            feats_dict[label].append(feature)
            idxs_dict[label].append(idx)




        # GD_dict = calculate_GD(feats_label, 0, class_per_task)

        select_idx_all = []
        for key in feats_dict:

            feats_per_classes = np.array(feats_dict[key])
            fully_GD_mean = torch.mean(torch.from_numpy(feats_per_classes).float(),axis=0)
            idxs_per_classes = np.array(idxs_dict[key])

            best_sim = -100000
            best_choice = 0
            for j in range(len(feats_per_classes)):
                select_feats_per_classes = torch.from_numpy(feats_per_classes[j])

                select_idxs_per_classes = idxs_per_classes[j]

                sim = torch.dot(select_feats_per_classes.float(), fully_GD_mean)

                if sim>best_sim:
                    best_sim = sim
                    best_choice = copy.copy(select_idxs_per_classes)
            select_idx_all.append(best_choice)

        

        return np.array(select_idx_all)




