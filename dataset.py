import os
import random

import cv2
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image, ImageOps


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img_LAB = Image.open(filepath).convert('LAB')
    return img, img_LAB


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_LAB, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_LAB = img_LAB.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    #targetLAB = targetLAB.crop((iy, ix, iy + ip, ix + ip))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_LAB, img_tar, info_patch


def augment(img_in, img_LAB, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_LAB = ImageOps.flip(img_LAB)
        img_tar = ImageOps.flip(img_tar)
        #targetLAB = ImageOps.flip(targetLAB)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_LAB = ImageOps.mirror(img_LAB)
            img_tar = ImageOps.mirror(img_tar)
            #targetLAB = ImageOps.mirror(targetLAB)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_LAB = img_LAB.rotate(180)
            img_tar = img_tar.rotate(180)
            #targetLAB = targetLAB.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_LAB, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, label_dir, data_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.label_path = label_dir 
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        _, file = os.path.split(self.data_filenames[index])

        k = random.randint(0,3)
        if k == 0:
            label_filenames = self.label_path + '/O/' + file
        if k == 1 :
            label_filenames = self.label_path + '/S/' + file
        if k == 2 :
            label_filenames = self.label_path + '/C/' + file
        if k == 3 :
            label_filenames = self.label_path + '/G/' + file


        target, targetLAB = load_img(label_filenames)
        input, inputLAB = load_img(self.data_filenames[index])
        # # 将input转化色域
        # inputLAB = cv2.cvtColor(input, cv2.COLOR_RGB2Lab)
        # targetLAB = cv2.cvtColor(target, cv2.COLOR_RGB2Lab)

        input = input.resize((512, 512), resample=Image.BICUBIC)
        target = target.resize((512, 512), resample=Image.BICUBIC)
        inputLAB = inputLAB.resize((512, 512), resample=Image.BICUBIC)
        #targetLAB = targetLAB.resize((512, 512), resample=Image.BICUBIC)


        input, inputLAB, target,_ = get_patch(input, inputLAB, target, self.patch_size)
        
        if self.data_augmentation:
            input, inputLAB, target, _ = augment(input, inputLAB, target)
        
        if self.transform:
            input = self.transform(input)
            inputLAB = self.transform(inputLAB)
            target = self.transform(target)
            #targetLAB = self.transform(targetLAB)

        return input, inputLAB, target, file

    def __len__(self):
        return len(self.data_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames

        self.transform = transform

    def __getitem__(self, index):
        input, inputLAB = load_img(self.data_filenames[index])

        label, labelLAB = load_img(self.label_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        (ih, iw) = input.size
        dh = ih % 8
        dw = iw % 8
        new_h, new_w = ih - dh, iw - dw

        input = input.resize((new_h, new_w))
        label = label.resize((new_h, new_w))
        inputLAB = inputLAB.resize((new_h, new_w))
        #labelLAB = labelLAB.resize((new_h, new_w))

        if self.transform:
            input = self.transform(input)
            label = self.transform(label)
            inputLAB = self.transform(inputLAB)
            #labelLAB = self.transform(labelLAB)

            
        return input, inputLAB, label, file
      
    def __len__(self):
        return len(self.data_filenames)


