import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from ..base_dataset import Datum, DatasetBase


id_to_name = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}


@DATASET_REGISTRY.register()
class CIFAR100_full(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar100"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, "train")
        test_dir = osp.join(self.dataset_dir, "test")


        train_x = self._read_data_train(train_dir)
        # print(train_x[0])
        # exit()
        test = self._read_data_test(test_dir)


        super().__init__(train_x=train_x, val=test, test=test)

    # def _read_data_train(self, data_dir, num_labeled, val_percent):
    #     class_names = listdir_nohidden(data_dir)
    #     class_names.sort()
    #     num_labeled_per_class = num_labeled / len(class_names)
    #     items_x, items_u, items_v = [], [], []

    #     for label, class_name in enumerate(class_names):
    #         class_dir = osp.join(data_dir, class_name)
    #         imnames = listdir_nohidden(class_dir)

    #         # Split into train and val following Oliver et al. 2018
    #         # Set cfg.DATASET.VAL_PERCENT to 0 to not use val data
    #         num_val = math.floor(len(imnames) * val_percent)
    #         imnames_train = imnames[num_val:]
    #         imnames_val = imnames[:num_val]

    #         # Note we do shuffle after split
    #         random.shuffle(imnames_train)

    #         for i, imname in enumerate(imnames_train):
    #             impath = osp.join(class_dir, imname)
    #             item = Datum(impath=impath, label=label)

    #             if (i + 1) <= num_labeled_per_class:
    #                 items_x.append(item)

    #             else:
    #                 items_u.append(item)

    #         for imname in imnames_val:
    #             impath = osp.join(class_dir, imname)
    #             item = Datum(impath=impath, label=label)
    #             items_v.append(item)

    #     return items_x, items_u, items_v


    def _read_data_train(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        # num_labeled_per_class = num_labeled / len(class_names)
        items_x = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)


            imnames_train = imnames


            # Note we do shuffle after split
            random.shuffle(imnames_train)

            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label,classname=id_to_name[label])
                items_x.append(item)


        return items_x

    def _read_data_test(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
   
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)
                items.append(item)
           

        return items
    

@DATASET_REGISTRY.register()
class CUB200_few(DatasetBase):
    """CIFAR10 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar100"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, "train")
        test_dir = osp.join(self.dataset_dir, "test")


        train_x = self._read_data_train(train_dir)
        # print(train_x[0])
        # exit()
        test = self._read_data_test(test_dir)


        super().__init__(train_x=train_x, val=test, test=test)

    # def _read_data_train(self, data_dir, num_labeled, val_percent):
    #     class_names = listdir_nohidden(data_dir)
    #     class_names.sort()
    #     num_labeled_per_class = num_labeled / len(class_names)
    #     items_x, items_u, items_v = [], [], []

    #     for label, class_name in enumerate(class_names):
    #         class_dir = osp.join(data_dir, class_name)
    #         imnames = listdir_nohidden(class_dir)

    #         # Split into train and val following Oliver et al. 2018
    #         # Set cfg.DATASET.VAL_PERCENT to 0 to not use val data
    #         num_val = math.floor(len(imnames) * val_percent)
    #         imnames_train = imnames[num_val:]
    #         imnames_val = imnames[:num_val]

    #         # Note we do shuffle after split
    #         random.shuffle(imnames_train)

    #         for i, imname in enumerate(imnames_train):
    #             impath = osp.join(class_dir, imname)
    #             item = Datum(impath=impath, label=label)

    #             if (i + 1) <= num_labeled_per_class:
    #                 items_x.append(item)

    #             else:
    #                 items_u.append(item)

    #         for imname in imnames_val:
    #             impath = osp.join(class_dir, imname)
    #             item = Datum(impath=impath, label=label)
    #             items_v.append(item)

    #     return items_x, items_u, items_v


    def _read_data_train(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        # num_labeled_per_class = num_labeled / len(class_names)
        items_x = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)


            imnames_train = imnames


            # Note we do shuffle after split
            random.shuffle(imnames_train)

            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label,classname=id_to_name[label])
                items_x.append(item)


        return items_x

    def _read_data_test(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
   
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)
                items.append(item)
           

        return items


# @DATASET_REGISTRY.register()
# class CIFAR100(CIFAR10):
#     """CIFAR100 for SSL.

#     Reference:
#         - Krizhevsky. Learning Multiple Layers of Features
#         from Tiny Images. Tech report.
#     """

#     dataset_dir = "cifar100"

#     def __init__(self, cfg):
#         super().__init__(cfg)
