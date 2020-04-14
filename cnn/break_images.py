
from PIL import Image
import os
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from torchvision.datasets import VisionDataset
import cv2
import torchvision.transforms as transforms

break_height_width = 32
jump = 1024+512
root_main='../data/mnt/d/SIDD_Medium_Srgb/Data/'
for root, directories, filenames in os.walk(root_main):
    for filename in filenames:
        print(filename)
        if("NOISY" in filename or "DS_Store" in filename):
            continue;
        filename_NOISY =filename.replace("GT" ,"NOISY" ,1)
        label_file = os.path.join(root, filename)
        input_file = os.path.join(root ,filename_NOISY)
        print("input_file:  " +input_file)
        print("label_file:  " +label_file)
        img = Image.open(input_file).convert('RGB')
        target = Image.open(label_file).convert('RGB')
        width, height = img.size
        current_start_height = 0
        current_start_width = 0
        count = 1
        while(current_start_height+jump<height):
            while(current_start_width+jump<width):
                left = current_start_width
                right = current_start_width+break_height_width
                top = current_start_height
                bottom = current_start_height+break_height_width
                im1 = img.crop((left, top, right, bottom))
                target1 = target.crop((left, top, right, bottom))
                filenames = filename.split(".")
                filename_start = filenames[0]
                filename_end = filenames[1]
                filename_new = filename_start+"_"+str(count)+"."+filename_end

                filenames_NOISY = filename_NOISY.split(".")
                filename_start_NOISY = filenames_NOISY[0]
                filename_end_NOISY = filenames_NOISY[1]
                filename_NOISY_new = filename_start_NOISY + "_" + str(count) + "." + filename_end_NOISY
                #im1.show()
                #target1.show()

                im1.save(os.path.join(root, filename_NOISY_new))
                target1.save(os.path.join(root, filename_new ))
                count = count+1
                current_start_width = current_start_width+jump

            current_start_height = current_start_height+jump
            current_start_width=0
        os.remove(label_file)
        os.remove(input_file)
