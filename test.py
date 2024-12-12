# # import cv2
# # from config import val_loader, G
# # import torchvision.utils as vutils
# #
# #
# # if __name__ == '__main__':
# #     imgs = next(iter(val_loader))
# #     out = G(imgs['profile'])
# #     print(imgs['front'][:5].shape)
# #     vutils.save_image(out, './test.png', nrow=5)
#
#
import os
import face_recognition
from scipy.spatial.distance import cosine
import torchvision
from pytorch_fid import fid_score
import torchvision.transforms as transforms
from models.model_with_attention import Generator, Discriminator_Attention, Discriminator_mark


G = Generator()
D_m = Discriminator_mark()
D_a = Discriminator_Attention()

total_params1 = sum(p.numel() for p in G.parameters())
total_params2 = sum(p.numel() for p in D_m.parameters())
total_params3 = sum(p.numel() for p in D_a.parameters())
print(total_params1, total_params2, total_params3)
# calculate parameters size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


sizeG = get_model_size(G)
sizeD_m = get_model_size(D_m)
sizeD_a = get_model_size(D_a)
print(f'G：{sizeG:.2f}MB')
print(f'D_m：{sizeD_m:.2f}MB')
print(f'D_a：{sizeD_a:.2f}MB')


# 准备真实数据分布和生成模型的图像数据
real_images = './path/real_images/folder'
generated_images = './path/generated_images/folder'
# Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)
# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

def calculate_fid(real_images_path, generated_images_path):
    fid = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], batch_size=50, device='cuda', dims=2048)
    return fid


print('FID value:', calculate_fid(real_images, generated_images))


def get_cos_similarity(rea_img_path, generate_img_path):
    image_path_list1 = os.listdir(rea_img_path)
    image_path_list2 = os.listdir(generate_img_path)
    img_num = len(image_path_list1)
    cos_similarity = 0.0
    for i in range(img_num):
        real_img_path, gene_img_path = image_path_list1[i], image_path_list2[i]
        image1 = face_recognition.load_image_file(real_img_path)
        image2 = face_recognition.load_image_file(gene_img_path)
        face_encoding1 = face_recognition.face_encodings(image1)
        face_encoding2 = face_recognition.face_encodings(image2)

        # 计算余弦相似度
        similarity = 1 - cosine(face_encoding1, face_encoding2)
        cos_similarity += similarity
    cos_similarity /= img_num
    return cos_similarity





