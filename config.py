import argparse
import torch
from tools import utils
import torch.optim as optimizer
from torch.nn import L1Loss, CosineSimilarity, MSELoss
from torch.utils.data import DataLoader
from datasets import ImageData, transforms
from models.model_with_attention import Generator, Discriminator_Attention, Discriminator_mark

__all__ = ['torch', 'args', 'extract_net', 'G', 'D_attention', 'D_landmark', 'train_loader', 'val_loader',
           'optimizer_G', 'optimizer_D_attention', 'optimizer_D_landmark', 'L1', 'mse', 'cosine', 'total_Var']

parser = argparse.ArgumentParser(description='PG_GAN_With_Attention')
parser.add_argument('--profile_train_dir', type=str, default='./data/train_data/profile_face')
parser.add_argument('--front_train_dir', type=str, default='./data/train_data/frontal_face')
parser.add_argument('--profile_test_dir', type=str, default='./data/test_data/profile_face')
parser.add_argument('--front_test_dir', type=str, default='./data/test_data/frontal_face')
parser.add_argument('--extract_net', type=str, default='./tools/InceptionResnet.mdl')

parser.add_argument('--modelout', type=str, default='./modelout')
parser.add_argument('--imgout', type=str, default='./imgout')

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--resume_model', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)

parser.add_argument('--lambda_L1', type=float, default=10)
parser.add_argument('--lambda_adv1', type=float, default=0.2)
parser.add_argument('--lambda_adv2', type=float, default=0.1)
parser.add_argument('--lambda_id', type=float, default=0.05)
parser.add_argument('--lambda_tv', type=float, default=1e-4)
parser.add_argument('--lambda_gp', type=float, default=10)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

# s随机种子
torch.manual_seed(1234)

extract_net = torch.load(args.extract_net).to(args.device)
utils.set_requires_grad(extract_net, False)

# initial net
G = Generator().to(args.device)
D_attention = Discriminator_Attention().to(args.device)
D_landmark = Discriminator_mark().to(args.device)


if args.resume_model:
    utils.resume_model(G, 'G_param', args.modelout, args.start_epoch)
    utils.resume_model(D_attention, 'D_attention_param', args.modelout, args.start_epoch)
    utils.resume_model(D_landmark, 'D_landmark_param', args.modelout, args.start_epoch)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


train_dataset = ImageData(args.profile_train_dir, args.front_train_dir, transform=transform)
val_dataset = ImageData(args.profile_test_dir, args.front_test_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

optimizer_G = optimizer.Adam(G.parameters(), lr=args.lr)
optimizer_D_attention = optimizer.Adam(D_attention.parameters(), lr=args.lr)
optimizer_D_landmark = optimizer.Adam(D_landmark.parameters(), lr=args.lr)

L1 = L1Loss().to(args.device)
mse = MSELoss().to(args.device)
cosine = CosineSimilarity(dim=1).to(args.device)


# 全变分损失
def total_Var(gen_f):
    genf_tv = torch.mean(torch.abs(gen_f[:, :, :-1, :] - gen_f[:, :, 1:, :])) + torch.mean(
        torch.abs(gen_f[:, :, :, :-1] - gen_f[:, :, :, 1:]))

    return genf_tv

