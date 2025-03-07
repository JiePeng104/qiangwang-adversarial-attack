import argparse
import os
import sys

sys.path.append('../')
sys.path.append(os.getcwd())

import model.models as m

from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from pytorch_msssim import ms_ssim  # 用于计算 MS-SSIM
from dct import *
from torch.autograd import Variable as V

# 导入赛方模型
from model.arcface.backbones import get_model


def load_surrogate_model(device):
    """ Load white-box and black-box models

    :return:
        face recognition and attribute recognition models
    """
    # Load pretrain white-box FR surrogate model
    fr_model = m.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('model/ir152.pth'))
    fr_model.to(device)
    fr_model.eval()

    # 赛方模型
    sf_model_list = []
    sf_model = get_model('r50').eval()
    sf_model.load_state_dict(torch.load('model/checkpoint/model.pt', map_location=device), strict=False)
    sf_model.to(device)
    sf_model_list.append(sf_model)
    return fr_model, sf_model_list


def infer_fr_model(attack_img, victim_img, fr_model):
    """ Face recognition inference

    :param attack_img:
            attacker face image
    :param victim_img:
            victim face image
    :param fr_model:
            face recognition model
    :return:
        feature representations for the attacker and victim face images
    """
    # 人脸识别提取的特征
    attack_img_feat = fr_model(attack_img)
    victim_img_feat = fr_model(victim_img)
    return attack_img_feat, victim_img_feat


def infer_sf_model(attack_img, victim_img, sf_model_list):
    """ Face recognition inference

    :param attack_img:
            attacker face image
    :param victim_img:
            victim face image
    :param fr_model:
            face recognition model
    :return:
        feature representations for the attacker and victim face images
    """
    loss_list = []
    for model in sf_model_list:
        # 人脸识别提取的特征
        attack_img_feat = model(attack_img)
        victim_img_feat = model(victim_img)
        loss = 1 - cos_simi(attack_img_feat, victim_img_feat)
        loss_list.append(loss)
    return sum(loss_list) / len(loss_list)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def Spectrum_Simulation_Attack(config, image1, min, max, model):
    """
    The attack algorithm of our proposed Spectrum Simulate Attack
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation
    :param max: the max the clip operation
    :return: the adversarial images
    """
    image_width = 112
    num_iter = 10
    eps = config.eps / 255.0
    alpha = eps / num_iter
    x = image1.clone()
    grad = 0
    rho = config.rho  # rate
    N = config.N  # 变换数
    sigma = config.sigma

    for i in range(num_iter):
        noise = 0
        for n in range(N):
            gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)

            feature_adv = model(x_idct)
            feature_orig = model(image1)
            similarity_loss = F.cosine_similarity(feature_adv, feature_orig, dim=-1).mean()
            loss = -similarity_loss

            loss.backward()
            noise += x_idct.grad.data
        noise = noise / N

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def target_Spectrum_Simulation_Attack(config, image1, image2, min, max, model, sf_model_list):
    """
    The attack algorithm of our proposed Spectrum Simulate Attack
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation
    :param max: the max the clip operation
    :return: the adversarial images
    """
    image_width = 112
    num_iter = 10
    eps = config.eps / 255.0
    alpha = eps / num_iter
    x = image1.clone()
    grad = 0
    rho = config.rho  # rate
    N = config.N  # 变换数
    sigma = config.sigma

    for i in range(num_iter):
        noise = 0
        for n in range(N):
            gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)

            feature_adv = model(x_idct)
            feature_orig = model(image2)
            # loss = F.cross_entropy(output_v3, gt)
            loss = F.cosine_similarity(feature_adv, feature_orig, dim=-1).mean()
            # loss = 0.
            # 验证攻击图像是否成功
            for model1 in sf_model_list:
                # 人脸识别提取的特征
                attack_feat = model1(x_idct)
                victim_feat = model1(image2)
                loss1 = F.cosine_similarity(attack_feat, victim_feat, dim=-1).mean()
            l = loss + loss1
            l.backward()
            noise += x_idct.grad.data
        noise = noise / N

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)

    # 验证攻击图像是否成功
    for model in sf_model_list:
        # 人脸识别提取的特征
        attack_img_feat = model(x)
        victim_img_feat = model(image2)
        simi = (torch.sum(torch.mul(attack_img_feat, victim_img_feat), dim=1)
                / attack_img_feat.norm(dim=1) / victim_img_feat.norm(dim=1))
    logging.info("攻击成功数量: %d" % (torch.sum(simi > 0.3).item()))
    # 攻击指标
    perturbation = x - image2
    for b in range(perturbation.shape[0]):  # 逐样本
        logging.info(f'样本{b}')
        for c in range(perturbation.shape[1]):  # 逐通道
            # 无穷范数
            linf_norm = torch.norm(perturbation[b][c].view(1, -1), p=float('inf'), dim=1)
            # 2范数
            l2_norm = torch.norm(perturbation[b][c].view(1, -1), p=2, dim=1)
            # 0范数
            l0_norm = torch.norm(perturbation[b][c].view(1, -1), p=0, dim=1)
            logging.info(f"通道：{c}, {linf_norm.item()}, {l2_norm.item()}, {l0_norm.item()}")
        # 结构相似性
        ms_ssim_value = ms_ssim(x[b].unsqueeze(0), image1[b].unsqueeze(0), data_range=1.0, win_size=7).item()
        logging.info(f'ms_ssim: {ms_ssim_value}')
    return x.detach()


def spectrum_attack(config):
    mode = int(config.mode)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    fr_model, sf_model_list = load_surrogate_model(device)

    if mode == 0:  # no target attack scene
        pairs_path = "data/no_target/pairs.txt"
        clean_imgs_path = 'data/no_target/images'
    elif mode == 1:  # target attack scene
        pairs_path = "data/target/pairs.txt"
        clean_imgs_path = 'data/target/images'
        target_imgs_path = 'data/target/TargetPerson'
    else:
        RuntimeError('Enter the mode:0-no target,1-target')

    # results_base_path = 'datasets/results/target'
    transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    if mode == 0:
        dataset = LFW(clean_imgs_path, '', pairs_path, transform=None)
    else:
        dataset = LFWDataset(target_imgs_path, clean_imgs_path, pairs_path, transform=transform)

    # dataset = LFW(clean_imgs_path, target_imgs_path, pairs_path, transform=transform)
    images_num = len(dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    logging.info('train begin!')
    # 新建攻击路径
    adv_dir = f'result_data/advimages/{config.saving_dir}/'
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir, exist_ok=True)
    # 展开对抗攻击
    i = 0
    if mode == 1:
        for left, right, right_name in tqdm(dataloader):
            image_left, image_right = left.to(device), right.to(device)
            images_min = clip_by_tensor(image_right - 32.0 / 255.0, -1.0, 1.0)
            images_max = clip_by_tensor(image_right + 32.0 / 255.0, -1.0, 1.0)
            adv_attack_img = target_Spectrum_Simulation_Attack(config, image_right, image_left, images_min, images_max,
                                                               fr_model,
                                                               sf_model_list)
            for n, name in enumerate(right_name):
                data_name = name.split('/')[0]
                file_name = name.split('/')[1]
                save_dir = adv_dir + data_name
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = save_dir + '/' + file_name
                save_adv_img(adv_attack_img[n].squeeze(0).cpu(), save_path, config)
                logging.info(f"Save adversarial image to - {save_path}")
    else:
        unnormalize = transforms.Compose([
            transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])
        ])
        for index in tqdm(range(images_num)):
            img_left = os.path.join(dataset.root, dataset.nameLs[index])
            img_right = os.path.join(dataset.root, dataset.nameRs[index])
            image1 = Image.open(img_left)
            image1 = transform(image1).unsqueeze(0)  # 拓展维度
            image1 = image1.cuda()

            image2 = Image.open(img_right)
            image2 = transform(image2).unsqueeze(0)  # 拓展维度
            image2 = image2.cuda()

            images_min = clip_by_tensor(image1 - config.eps / 255.0, -1.0, 1.0)
            images_max = clip_by_tensor(image1 + config.eps / 255.0, -1.0, 1.0)
            adv_img = Spectrum_Simulation_Attack(config, image1, images_min, images_max, sf_model_list[0])

            adv_img_path = os.path.join(adv_dir, dataset.nameLs[index])
            os.makedirs(os.path.dirname(adv_img_path), exist_ok=True)  # 创建目录结构
            # 保存对抗样本
            # adv_image1 = unnormalize(adv_image1)  # 反归一化
            to_pil = ToPILImage()
            adv_image1 = unnormalize(adv_img)  # 反归一化
            adv_image_pil = to_pil(adv_image1.squeeze(0))  # 转换为PIL图像
            adv_image_pil.save(adv_img_path)

if __name__ == '__main__':
    # 从控制台读入参数
    parser = argparse.ArgumentParser(description='code from Hidden Face')
    # Add command line parameters

    parser.add_argument('--mode', type=str, help='Enter the mode:0-no target,1-target', default=0)
    parser.add_argument('--eps', type=float, default=32, help="epsilon of L_inf")
    parser.add_argument('--num_iter', type=int, default=10, help="number of iteration of attacks")
    parser.add_argument('--rho', type=float, default=0.6, help="tuning factor")
    parser.add_argument('--N', type=int, default=30, help="number of different frequency ")
    parser.add_argument('--sigma', type=float, default=20,
                        help="sigma of gaussian noise during training (std^2=sigma/255)")
    parser.add_argument('--saving_dir', type=str, default='no_target', help='result dir name')

    args, unknown = parser.parse_known_args()

    logging.basicConfig(filename='test/train.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    spectrum_attack(args)
