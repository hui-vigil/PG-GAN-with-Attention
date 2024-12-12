import os
import numpy as np
import torch.nn as nn
import cv2
import torch
from torch.autograd import grad
from mtcnn import MTCNN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import torchvision.utils as vutils


detector = MTCNN()
def get_mark(path, mode='real'):
    if mode == 'front':
        img = cv2.imread(path, 0)
        for i in range(128):
            for j in range(128):
                if img[i, j] == 255:
                    img[i, j] = 1
        img = img * 1.0
        face_mask = gaussian_filter(img, sigma=2)
        face_mask = torch.as_tensor(face_mask, dtype=torch.float32).unsqueeze(0)
        return face_mask
    img = cv2.imread(path)
    face_mask = np.zeros_like(img[:, :, 0], dtype=np.float32)
    faces = detector.detect_faces(img)
    for face in faces:
        for x, y in face['keypoints'].values():
            face_mask[y, x] = 1.0  # tranpose x, y in face_mask
    face_mask = gaussian_filter(face_mask, sigma=2)
    face_mask = torch.from_numpy(face_mask).unsqueeze(0)

    return face_mask


def show_img(tensor):
    img = transforms.ToPILImage(tensor)  # tensor 转PIL图片
    plt.imshow(img, camp='gray')
    plt.show()


def set_requires_grad(module, b):
    for param in module.parameters():
        param.requires_grad = b


def save_model(model_state_dict, dirname, epoch):
    # if type(model).__name__ == nn.DataParallel.__name__:
    #     model = model.module
    torch.save(model_state_dict,
               f'{dirname}/epoch_{epoch}_checkpoint.ckpt')


def resume_model(model, dict_name, dirname, epoch, strict=True):
    if type(model).__name__ == nn.DataParallel.__name__:
        model = model.module
    path = f'{dirname}/epoch_{epoch}_checkpoint.pth'
    if os.path.exists(path):
        state = torch.load(path)[dict_name]
        model.load_state_dict(state, strict=strict)
    else:
        raise FileNotFoundError(f'Not found {path}')


def get_gp_loss(real_output, fake_output, Interpolates, D, device):
    loss_no_GP = -torch.mean(real_output) + torch.mean(fake_output)
    out = D(Interpolates)
    out_grad = grad(outputs=out, inputs=Interpolates,
                    grad_outputs=torch.ones(out.size()).to(device),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True)[0].view(out.size(0), -1)
    gradient_penalty = torch.mean((torch.norm(out_grad, p=2) - 1) ** 2)
    gp_loss = loss_no_GP + 10 * gradient_penalty
    return gp_loss


def segment(inp):  # 输入图片局部分割
    # 128x128
    # face：22:22+88, 19:19+91
    # eye：32:32+26, 26:26+77
    # nose：39:39+41, 50:50+29
    # mouth：76:76+23, 44:44+41
    eyes = inp[:, :, 32:32 + 26, 26:26 + 77]
    nose = inp[:, :, 39:39 + 41, 50:50 + 29]
    mouth = inp[:, :, 76:76 + 23, 44:44 + 41]
    face = inp[:, :, 22:22 + 88, 19:19 + 91]

    return eyes, nose, mouth, face


def save_imgs(G, data_faces, outimg_dir, epoch):
    G.eval()
    with torch.no_grad():
        img128, img64, img32 = G(data_faces['profile'], data_faces['real_lm'], data_faces['target_lm'])

        img_profile = vutils.make_grid(data_faces['profile'], nrow=1, normalize=False, pad_value=255)
        syn_front = vutils.make_grid(img128.detach(), nrow=1, normalize=False, pad_value=255)
        true_front = vutils.make_grid(data_faces['front'], nrow=1, normalize=False, pad_value=255)
        image_grid = torch.cat([img_profile, syn_front, true_front], dim=2)
        vutils.save_image(image_grid, f'{outimg_dir}/epoch_{epoch}_test.png', padding=4)


# 得到训练数据和生成数据图片
def sample_images(G, train_loader, val_loader, epoch, device, outimg_dir):

    """Saves a generated sample from the test set"""
    val_imgs = next(iter(val_loader))  # 得到一批数据
    train_imgs = next(iter(train_loader))
    for key in val_imgs.keys():
        val_imgs[key] = val_imgs[key][:5].to(device)
    for key in train_imgs.keys():
        train_imgs[key] = train_imgs[key][:5].to(device)

    # test_img
    save_imgs(G, val_imgs, outimg_dir, epoch)

    # train_img
    save_imgs(G, train_imgs, outimg_dir, epoch)
