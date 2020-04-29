from PIL import Image
import os
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from torchvision.datasets import VisionDataset
import cv2
import torchvision.transforms as transforms

class DENOISE_DATASET(VisionDataset):
    def __init__(self, root, train_folder,label_folder,train=True, transform=None, target_transform=None):

        super(DENOISE_DATASET, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.data = []
        self.targets = []
        count = 0
        for root, directories, filenames in os.walk(root):
            for filename in filenames:
                if("NOISY" in filename):
                    continue
                filename_NOISY=filename.replace("GT","NOISY",1)
                label_file = os.path.join(root, filename)
                input_file = os.path.join(root,filename_NOISY)
                # print("train_file: "+input_file)
                # print("label_file: "+label_file)
                self.data.append(input_file)
                self.targets.append(label_file)
                count = count+1
        print("Total Training Samples: "+str(count))
        # train_files = [join(train_folder,f) for f in listdir(train_folder) if isfile(join(train_folder, f))]
        # label_files = [join(label_folder,f) for f in listdir(label_folder) if isfile(join(label_folder, f))]
        # train_files.sort()
        # label_files.sort()

        # if(len(train_files) != len(label_files)):
        #         sys.exit(-1)

        # now load the picked numpy arrays
        # for train_file,test_file in zip(train_files,label_files):
        #     self.data.append(train_file)
        #     self.targets.append(test_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        train_file, test_file = self.data[index], self.targets[index]
        img = Image.open(train_file).convert('RGB')
        target = Image.open(test_file).convert('RGB')
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        print("Img 1 Shape")
        print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        print("Img Shape")
        print(img.shape)
        return img, target

    def __len__(self):
        return len(self.data)


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

