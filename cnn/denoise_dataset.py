from torchvision.datasets import VisionDataset
from PIL import Image
import os
import sys
import numpy as np
from os import listdir
from os.path import isfile, join

class DENOISE_DATASET(VisionDataset):
    def __init__(self, root, train_folder,test_folder,train=True, transform=None, target_transform=None):

        super(DENOISE_DATASET, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.data = []
        self.targets = []

        train_files = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
        test_files = [f for f in listdir(test_folder) if isfile(join(test_folder, f))]
        train_files.sort()
        test_files.sort()

        if(len(train_files) != len(test_files)):
                sys.exit(-1)

        # now load the picked numpy arrays
        for train_file,test_file in zip(train_files,test_files):
            train_img = Image.open(train_file).convert('RGBA')
            test_img = Image.open(test_file).convert('RGBA')
            train_arr = np.array(train_img)
            test_arr = np.array(test_img)
            self.data.append(train_arr)
            self.targets.append(test_arr)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

