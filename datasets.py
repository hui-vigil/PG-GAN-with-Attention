from torch.utils.data import Dataset
from PIL import Image
import os
from tools.utils import get_mark, transforms


front_mark_path = './tools/mask128.png'
class ImageData(Dataset):
    def __init__(self, profile_dir, front_dir, transform=None):
        super(ImageData, self).__init__()
        self.profile_dir = profile_dir
        self.front_dir = front_dir
        self.profile_faces = os.listdir(profile_dir)
        self.front_faces = os.listdir(front_dir)
        self.transform = transform
        self.front_mark = get_mark(front_mark_path, 'front')
        self.trans_to_64 = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.trans_to_32 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.profile_faces)

    def __getitem__(self, index):
        data = dict()
        img_name = self.profile_faces[index]
        token = img_name.split('_')
        front_img_name = '_'.join(token[:3])+'_051_'+token[4]+'_crop_128.png'
        profile_img_path = os.path.join(self.profile_dir, img_name)
        front_img_path = os.path.join(self.front_dir, front_img_name)
        with Image.open(profile_img_path) as profile_face:
            data['profile'] = self.transform(profile_face)
        with Image.open(front_img_path) as frontal_face:
            data['front'] = self.transform(frontal_face)

        data['real_mark'] = get_mark(profile_img_path, 'real')
        data['front_mark'] = self.front_mark
        data['img64'] = self.trans_to_64(frontal_face)
        data['img32'] = self.trans_to_32(frontal_face)
        return data







