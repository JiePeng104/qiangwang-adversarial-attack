# import cv2
# import torch
# import torch.nn.functional as F

# def load_img(img_path, config):
#     """ Read image from path and convert it to torch tensor

#     Args:
#         :param img_path:
#             the path to the input face image
#         :param config:
#             attacking configurations

#     :return:
#         the processed face image torch tensor object
#     """
#     m = config.dataset['mean']
#     s = config.dataset['std']
#     input_size = tuple(config.dataset['input_size'])

#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
#     img = torch.from_numpy(img).to(torch.float32)
#     img = img.transpose(0, 2).transpose(1, 2).unsqueeze(0)
#     mean = torch.tensor(m).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#     std = torch.tensor(s).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#     img = (img - mean) / std
#     img = F.interpolate(img, size=input_size, mode='bilinear')
#     return img

# def save_adv_img(attack_img, save_path, config):
#     """ Save adversarial image to the distination path

#     Args:
#         :param attack_img:
#             adversarial face image
#         :param save_path:
#             the path to save the adversarial face image
#         :param config:
#             attacking configurations

#     :return:
#         None
#     """
#     input_size = tuple(config.dataset['input_size'])
#     m = config.dataset['mean']
#     s = config.dataset['std']
#     mean = torch.tensor(m).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#     std = torch.tensor(s).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

#     attack_img = F.interpolate(attack_img, size=input_size, mode='bilinear')
#     attack_img = (attack_img * mean + std) * 255
#     attack_img = attack_img[0].transpose(0, 1).transpose(1, 2)
#     attack_img = attack_img.numpy()
#     attack_img = cv2.cvtColor(attack_img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(save_path, attack_img)

# def cos_simi(emb_attack_img, emb_victim_img):
#     """ Calculate cosine similarity between two face features

#     Args:
#         :param emb_attack_img:
#             input feature representation for the attacker
#         :param emb_victim_img:
#             input feature representation for the victim

#     :return:
#         the cosine similarity of two features
#     """
#     return torch.mean(torch.sum(torch.mul(emb_victim_img, emb_attack_img), dim=1)
#                       / emb_victim_img.norm(dim=1) / emb_attack_img.norm(dim=1))

# def obtain_attacker_victim(config):
#     """ Obtain attackers and victims' image paths

#     Args:
#         :param config:
#             attacking configurations

#     :return:
#         the split path groups of attack and victim face images
#     """
#     dataset_dir = config.dataset['dataset_dir']
#     dataset_txt = config.dataset['dataset_txt']
#     attack_img_paths = []
#     victim_img_paths = []
#     with open(dataset_txt) as fin:
#         img_names = fin.readlines()
#         for idx, img_name in enumerate(img_names):
#             img_path = dataset_dir + '/' + img_name.strip()
#             if idx < 10:
#                 victim_img_paths.append(img_path)
#             else:
#                 attack_img_paths.append(img_path)

#     return attack_img_paths, victim_img_paths

# # 赛题
# import torch.utils.data as data
# import cv2
# import numpy as np
# import os
# import torch

# def img_loader(path):
#     try:
#         with open(path, 'rb') as f:
#             img = cv2.imread(path)
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, 2)
#             return img
#     except IOError:
#         print('Cannot load image ' + path)

# class LFW(data.Dataset):
#     def __init__(self, root, root_target, file_list, transform=None, loader=img_loader):

#         self.root = root
#         self.root_target = root_target
#         self.file_list = file_list
#         self.transform = transform
#         self.loader = loader
#         self.nameLs = []
#         self.nameRs = []
#         self.personLs = []
#         self.personRs = []
#         self.nameLsFile = []
#         self.nameRsFile = []
#         self.folds = []
#         self.flags = []

#         with open(file_list) as f:
#             pairs = f.read().splitlines()[0:]
#         for i, p in enumerate(pairs):
#             p = p.split(' ')
#             # no target
#             if len(p) == 3:
#                 nameL = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
#                 nameR = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
#                 self.personLs.append(p[0])
#                 self.personRs.append(p[0])
#                 nameLsFile = p[0] + '_' + '{:04}.png'.format(int(p[1]))
#                 nameRsFile = p[0] + '_' + '{:04}.png'.format(int(p[2]))
#                 fold = i // 600
#                 flag = 1
#             # target
#             elif len(p) == 4:
#                 #nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
#                 nameL = p[0] + '_' + '{:04}.png'.format(int(p[1]))
#                 nameR = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
#                 self.personLs.append(p[0])
#                 self.personRs.append(p[2])
#                 nameLsFile = p[0] + '_' + '{:04}.png'.format(int(p[1]))
#                 nameRsFile = p[2] + '_' + '{:04}.png'.format(int(p[3]))
#                 fold = i // 600
#                 flag = -1
#             self.nameLs.append(nameL)
#             self.nameRs.append(nameR)
#             self.nameLsFile.append(nameLsFile)
#             self.nameRsFile.append(nameRsFile)
#             self.folds.append(fold)
#             self.flags.append(flag)

#     def __getitem__(self, index):

#         img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
#         img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
#         imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

#         if self.transform is not None:
#             for i in range(len(imglist)):
#                 imglist[i] = self.transform(imglist[i])

#             imgs = imglist
#             return imgs
#         else:
#             imgs = [torch.from_numpy(i) for i in imglist]
#             return imgs

#     def __len__(self):
#         return len(self.nameLs)

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()


def load_img(img_path):
    """ Read image from path and convert it to torch tensor

    Args:
        :param img_path:
            the path to the input face image
        :param config:
            attacking configurations

    :return:
        the processed face image torch tensor object
    """
    # try:
    #     with open(img_path, 'rb') as f:
    #         img = cv2.imread(img_path)
    #         if len(img.shape) == 2:
    #             img = np.stack([img] * 3, 2)
    #         return img
    # except IOError:
    #     print('Cannot load image ' + img_path)
    # img = Image.open(img_path)
    # return img
    m = 0.5
    s = 0.5
    input_size = tuple([112, 112])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32)
    img = img.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    mean = torch.tensor(m).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(s).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    img = (img - mean) / std
    img = F.interpolate(img, size=input_size, mode='bilinear')
    return img


def save_adv_img(attack_img, save_path, config):
    """ Save adversarial image to the distination path

    Args:
        :param attack_img:
            adversarial face image
        :param save_path:
            the path to save the adversarial face image
        :param config:
            attacking configurations

    :return:
        None
    """
    # input_size = tuple(config.dataset['input_size'])
    # m = config.dataset['mean']
    # s = config.dataset['std']
    # mean = torch.tensor(m).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # std = torch.tensor(s).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # attack_img = F.interpolate(attack_img, size=input_size, mode='bilinear')
    # attack_img = (attack_img * mean + std) * 255
    # attack_img = attack_img[0].transpose(0, 1).transpose(1, 2)
    # attack_img = attack_img.numpy()
    # attack_img = cv2.cvtColor(attack_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, attack_img)
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])
    ])
    adv_image1 = unnormalize(attack_img)  # 反归一化   
    adv_image_pil = to_pil(adv_image1.squeeze(0).cpu())  # 转换为PIL图像并移至CPU
    adv_image_pil.save(save_path)


def cos_simi(emb_attack_img, emb_victim_img):
    """ Calculate cosine similarity between two face features

    Args:
        :param emb_attack_img:
            input feature representation for the attacker
        :param emb_victim_img:
            input feature representation for the victim

    :return:
        the cosine similarity of two features
    """
    return torch.mean(torch.sum(torch.mul(emb_victim_img, emb_attack_img), dim=1)
                      / emb_victim_img.norm(dim=1) / emb_attack_img.norm(dim=1))


def obtain_attacker_victim(config):
    """ Obtain attackers and victims' image paths

    Args:
        :param config:
            attacking configurations

    :return:
        the split path groups of attack and victim face images
    """
    dataset_dir = config.dataset['dataset_dir']
    dataset_txt = config.dataset['dataset_txt']
    attack_img_paths = []
    victim_img_paths = []
    with open(dataset_txt) as fin:
        img_names = fin.readlines()
        for idx, img_name in enumerate(img_names):
            img_path = dataset_dir + '/' + img_name.strip()
            if idx < 10:
                victim_img_paths.append(img_path)
            else:
                attack_img_paths.append(img_path)

    return attack_img_paths, victim_img_paths


# 赛题
import torch.utils.data as data
import cv2
import numpy as np
import os
import torch


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


def get_image(path):
    # image = Image.open(path)
    # transform = transforms.Compose([
    #     transforms.Resize([112,112]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=0.5, std=0.5)
    # ])
    # image = transform(image).unsqueeze(0) #拓展维度
    # # image = image.to(device)
    # return image
    img = Image.open(path)
    return img


class LFW(data.Dataset):
    def __init__(self, root, root_target, file_list, transform=None, loader=img_loader):

        self.root = root
        self.root_target = root_target
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.personLs = []
        self.personRs = []
        self.nameLsFile = []
        self.nameRsFile = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[0:]
        for i, p in enumerate(pairs):
            p = p.split(' ')
            # no target
            if len(p) == 3:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameR = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
                self.personLs.append(p[0])
                self.personRs.append(p[0])
                nameLsFile = p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameRsFile = p[0] + '_' + '{:04}.png'.format(int(p[2]))
                fold = i // 600
                flag = 1
            # target
            elif len(p) == 4:
                # nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameL = p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameR = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
                self.personLs.append(p[0])
                self.personRs.append(p[2])
                nameLsFile = p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameRsFile = p[2] + '_' + '{:04}.png'.format(int(p[3]))
                fold = i // 600
                flag = -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.nameLsFile.append(nameLsFile)
            self.nameRsFile.append(nameRsFile)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)


# 并行batch运行 + 有目标攻击
class LFWDataset(data.Dataset):
    def __init__(self, root_left, root_right, file_list, transform=None, loader=get_image):

        self.root_left = root_left
        self.root_right = root_right
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.personLs = []
        self.personRs = []
        self.nameLsFile = []
        self.nameRsFile = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[0:]
        for i, p in enumerate(pairs):
            p = p.split(' ')
            # no target
            if len(p) == 3:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameR = p[0] + '/' + p[0] + '_' + '{:04}.png'.format(int(p[2]))
                self.personLs.append(p[0])
                self.personRs.append(p[0])
                nameLsFile = p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameRsFile = p[0] + '_' + '{:04}.png'.format(int(p[2]))
                fold = i // 600
                flag = 1
            # target
            elif len(p) == 4:
                # nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameL = p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameR = p[2] + '/' + p[2] + '_' + '{:04}.png'.format(int(p[3]))
                self.personLs.append(p[0])
                self.personRs.append(p[2])
                nameLsFile = p[0] + '_' + '{:04}.png'.format(int(p[1]))
                nameRsFile = p[2] + '_' + '{:04}.png'.format(int(p[3]))
                fold = i // 600
                flag = -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.nameLsFile.append(nameLsFile)
            self.nameRsFile.append(nameRsFile)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):
        img_l = self.loader(os.path.join(self.root_left, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root_right, self.nameRs[index]))
        # 先不进行翻转
        imglist = [img_l, img_r]
        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])
            imgs = imglist
            imgs.append(self.nameRs[index])
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            imgs.append(self.nameRs[index])
            return imgs
        # 先不进行翻转
        # cvimglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]
        # imglist = []
        # if self.transform is not None:
        #     for i in range(len(cvimglist)):
        #         imglist[i] = Image.fromarray(cv2.cvtColor(imglist[i], cv2.COLOR_BGR2RGB))
        #         imglist[i] = self.transform(imglist[i])
        #     imgs = imglist
        #     return imgs
        # else:
        #     imgs = [torch.from_numpy(i) for i in imglist]
        # return imgs

    def __len__(self):
        return len(self.nameLs)
