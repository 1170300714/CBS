<div align="center">

# Class Balance Matters to Active Class-Incremental Learning 
#### Zitong Huang, Ze Chen, Yuanze Li, Bowen Dong, Erjin Zhou, Yong Liu, Rick Siow Mong Goh, Chun-Mei Feng, Wangmeng Zuo
Harbin Institute of Technology<br>
MEGVII Technology<br>
A*STAR<br>

</div>

# News
- **Dec-11-24**: Release the training code of experiments on [LP-DiF](https://github.com/1170300714/LP-DiF) (one of the three select CIL baseline).
- **Dec-10-24**: The paper has been preprinted in [Arxiv](https://arxiv.org/pdf/2412.06642).
- **July-16-24**: Our work is accepted by ACM MM 2024.

## Abstract
Few-Shot Class-Incremental Learning has shown remarkable efficacy in efficient learning new concepts with limited annotations. Nevertheless, the heuristic few-shot annotations may not always cover the most informative samples, which largely restricts the capability of incremental learner. We aim to start from a pool of large-scale unlabeled data and then annotate the most informative samples for incremental learning. Based on this purpose, this paper introduces the Active Class-Incremental Learning (ACIL). The objective of ACIL is to select the most informative samples from the unlabeled pool to effectively train an incremental learner, aiming to maximize the performance of the resulting model. Note that vanilla active learning algorithms suffer from class-imbalanced distribution among annotated samples, which restricts the ability of incremental learning. To achieve both class balance and informativeness in chosen samples, we propose Class-Balanced Selection (CBS) strategy. Specifically, we first cluster the features of all unlabeled images into multiple groups. Then for each cluster, we employ greedy selection strategy to ensure that the Gaussian distribution of the sampled features closely matches the Gaussian distribution of all unlabeled features within the cluster. Our CBS can be plugged and played into those CIL methods which are based on pretrained models with prompts tunning technique. Extensive experiments under ACIL protocol across five diverse datasets demonstrate that CBS outperforms both random selection and other SOTA active learning approaches.

## Datasets
The datasets of CUB-200, mini-ImageNet, DTD and Flowers-102 are uploaded to [google drive](https://drive.google.com/drive/folders/1Tdds6Ymqy4GmfrAHEyV7d3PzH1CbK0Qk?usp=drive_link). As for the CIFAR-100, it will automatically downloaded when running the followed experimental code.

## CBS for LP-DiF
###  Installation
1. The experiments on LP-DiF is based on the [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and the version of our other dependencies are following to them. Please follow their instruction to install. Thanks for their contribution.
2. Install other dependencies (It is recommended to use conda.):
    ```bash
    conda install pandas 
    conda install -c pytorch faiss-gpu
    ```

### Link to Datasets
We recommand using soft link to deploy the datasets. You can unzip the dataset downloaded above in any directory you want, and then create symbolic links to them from the root directory of LP-DiF.
```bash
cd CBS_LP-DiF
mkdir data
cd data
ln -s /path/to/cub200 ./CUB_200_2011
ln -s /path/to/miniImageNet ./miniimagenet
ln -s /path/to/dtd ./dtd
ln -s /path/to/Flowers102 ./Flowers102
cd ..
```


###  Training LP-DiF with CBS
Now that you have completed all the preparations, you can start training the model.
For example, if you want to running our CBS + unlabeled data on DTD dataset under ```B=100``` (equivalent to 5 rounds.), then you can run:

```bash
bash start_scripts/acil_scripts/start_dtd_wo_base_our_acil_distribution_kmeans_random_discard_greedy_add_pseudo.sh 5
```

if you want to running our CBS on Flowers102 dataset under ```B=20``` (equivalent to 1 rounds.), then you can run:

```bash
bash start_scripts/acil_scripts/start_flowers_wo_base_our_acil_distribution_kmeans_random_discard_greedy.sh 1
```
In  ```start_scripts/acil_scripts```, you can find more launch scripts of various AL method on various dataset, which are corresponding to experiments in our paper.

## TODO
- [x] Release training code of experiments on LP-DiF.
- [ ] Release training code of experiments on L2P.
- [ ] Release training code of experiments on L2P.