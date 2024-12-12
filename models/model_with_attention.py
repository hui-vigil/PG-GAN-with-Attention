from tools.utils import segment
from torch.autograd import grad
from models.layers import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #      0   1   2   3   4    5    6
        dim = [8, 16, 32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            conv(5, dim[3], kernel=7, stride=1, pad=3),  # 5*128*128 -> 64*128*128
            Residual_Block(dim[3], None, 3, 1, None)
        )
        self.conv2 = nn.Sequential(
            conv(dim[3], dim[3], kernel=5, stride=2, pad=2),  # 64*128*128 -> 64*64*64
            Residual_Block(dim[3], None, 3, 1, None)
        )
        self.conv3 = nn.Sequential(
            conv(dim[3], dim[4], kernel=3, stride=2, pad=1),  # 64*64*64 -> 128*32*32
            Residual_Block(dim[4], None, 3, 1, None)
        )
        self.conv4 = nn.Sequential(
            conv(dim[4], dim[5], kernel=3, stride=2, pad=1),  # 128*32*32 -> 256*16*16
            Residual_Block(dim[5], None, 3, 1, None)
        )
        self.conv5 = nn.Sequential(
            conv(dim[5], dim[6], kernel=3, stride=2, pad=1),  # 256*16*16 - > 512*8*8
            Residual_Block(dim[6], None, 3, 1, None)
        )
        self.fc1 = nn.Linear(512*8*8, dim[6])
        self.relu = nn.ReLU(inplace=True)
        self.maxout = nn.MaxPool1d(2)
        self.fc2 = nn.Linear(dim[5], dim[3]*8*8)  # B * 256 -> B * 4096
        # Decoder
        self.dc0_1 = deconv(dim[3], dim[2], 4, 4)  # 64*8*8 -> 32*32*32
        self.dc0_2 = deconv(dim[2], dim[1], 2, 2)  # 32*32*32 -> 16*64*64
        self.dc0_3 = deconv(dim[1], dim[0], 2, 2)  # 16*64*64 -> 8*128*128

        self.dc1 = nn.Sequential(
            deconv(dim[3]+dim[6], dim[6], 2, 2),  # 576*8*8 -> 512*16*16
            Residual_Block(dim[6], None, 3, 1, None), Residual_Block(dim[6], None, 3, 1, None)
        )
        self.dc2 = nn.Sequential(
            deconv(dim[5]+dim[6], dim[5], 2, 2),  # 768*16*16 -> 256*32*32
            Residual_Block(dim[5], None, 3, 1, None), Residual_Block(dim[5], None, 3, 1, None)
        )
        self.dc3 = nn.Sequential(
            deconv(dim[5]+dim[2]+dim[4]+3, dim[4], 2, 2),  # 419*32*32 -> 128*64*64
            Residual_Block(dim[4], None, 3, 1, None), Residual_Block(dim[4], None, 3, 1, None)
        )
        self.dc4 = nn.Sequential(
            deconv(dim[1]+dim[3]+dim[4]+3, dim[3], 2, 2),  # 211*64*64 -> 64*128*128
            Residual_Block(dim[3], None, 3, 1, None), Residual_Block(dim[3], None, 3, 1, None)
        )
        self.dc5 = nn.Sequential(
            deconv(dim[0]+dim[3]+dim[3]+3, dim[3], 3, 1, 1),  # 139*128*128 -> 64*128*128
            Residual_Block(dim[3], None, 3, 1, None)
        )
        self.conv6 = conv(dim[5], 3, 3, 1, 1)  # 256*32*32 -> 3*32*32
        self.conv7 = conv(dim[4], 3, 3, 1, 1)  # 128*64*64 -> 3*64*64
        self.conv8 = conv(dim[3], 3, 3, 1, 1)  # 64*128*128 -> 3*128*128
        self.self_att1 = Self_Attention(dim[3])  # 64
        self.self_att2 = Self_Attention(dim[4])  # 32
        self.self_att3 = Self_Attention(dim[5])  # 32
        self.self_att4 = Self_Attention(dim[4])  # 64

    def forward(self, img, real_mark, front_mark):
        b, _, _, _ = img.size()
        img_div_2 = nn.MaxPool2d(2)(img)  # 64x64
        img_div_4 = nn.MaxPool2d(2)(img_div_2)  # 32x32
        input_map = torch.cat([img, real_mark, front_mark], dim=1)
        conv1 = self.conv1(input_map)  # 64*128*128
        conv2 = self.conv2(conv1)  # 64*64*64
        conv2, attention1 = self.self_att1(conv2)
        conv3 = self.conv3(conv2)  # 128*32*32
        conv3, attention2 = self.self_att2(conv3)  # 128*32*32
        conv4 = self.conv4(conv3)  # 256*16*16
        conv5 = self.conv5(conv4)  # 512*8*8
        fc1_in = conv5.view(b, 512*8*8)  #
        fc1_out = self.fc1(fc1_in)  # 512
        fc1_relu = self.relu(fc1_out)
        fc1_relu = fc1_relu.unsqueeze(0)
        maxout = self.maxout(fc1_relu)[0]  # 256
        fc2_out = self.fc2(maxout)  # 64*8*8
        deconv_in = fc2_out.view(b, 64, 8, 8)
        dc01 = self.dc0_1(deconv_in)  # 32*32*32
        dc02 = self.dc0_2(dc01)  # 16*64*64
        dc03 = self.dc0_3(dc02)  # 8*128*128
        dc1 = self.dc1(torch.cat([conv5, deconv_in], dim=1))
        dc2 = self.dc2(torch.cat([conv4, dc1], dim=1))  # 256*32*32
        dc2, attention3 = self.self_att3(dc2)
        dc3 = self.dc3(torch.cat([img_div_4, conv3, dc01, dc2], dim=1))  # 128*64*64
        dc3, attention = self.self_att4(dc3)
        dc4 = self.dc4(torch.cat([img_div_2, conv2, dc02, dc3], dim=1))  # 64*128*128
        dc5 = self.dc5(torch.cat([img, conv1, dc03, dc4], dim=1))  # 64*128*128
        img32 = self.conv6(dc2)
        img64 = self.conv7(dc3)
        img128 = self.conv8(dc5)

        return img32, img64, img128

    @staticmethod
    def GLoss(Syn_F_GAN):
        Syn_F = Syn_F_GAN[0] + Syn_F_GAN[1] + Syn_F_GAN[2] + Syn_F_GAN[3] + Syn_F_GAN[4]
        loss = -Syn_F.mean() / 5

        return loss


class Discriminator_mark(nn.Module):
    def __init__(self):
        super(Discriminator_mark, self).__init__()
        self.model = nn.Sequential(
            conv(4, 64, 4, 2, 1),  # 64*64*64
            conv(64, 128, 4, 2, 1),  # 128*32*32
            conv(128, 256, 4, 2, 1),  # 256*16*16
            conv(256, 512, 4, 2, 1),  # 512*8*8
            conv(512, 512, 4, 1, 1),  # 512*7*7
            conv(512, 1, 4, 1, 1),  # 1*6*6
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Discriminator_Attention(nn.Module):
    def __init__(self):
        super(Discriminator_Attention, self).__init__()
        eta = 1e-2

        img_convlayers = [
            conv(3, 32, 3, 2, 1), nn.LeakyReLU(eta),
            conv(32, 64, 3, 2, 1), nn.LayerNorm([32, 32]), nn.LeakyReLU(eta),
            conv(64, 128, 3, 2, 1), nn.LayerNorm([16, 16]), nn.LeakyReLU(eta),
            conv(128, 256, 3, 2, 1), nn.LayerNorm([8, 8]), nn.LeakyReLU(eta),
            conv(256, 256, 3, 2, 1), nn.LayerNorm([4, 4]), nn.LeakyReLU(eta),
            Flatten(), nn.Linear(4096, 1)
        ]
        eyes_convlayers = [
            conv(3, 32, 3, 2, 1), nn.LeakyReLU(eta),
            conv(32, 64, 3, 2, 1), nn.LayerNorm([7, 20]), nn.LeakyReLU(eta),
            conv(64, 128, 3, 2, 1), nn.LayerNorm([4, 10]), nn.LeakyReLU(eta),
            conv(128, 256, 3, 2, 1), nn.LayerNorm([2, 5]), nn.LeakyReLU(eta),
            Flatten(), nn.Linear(2560, 1)
        ]
        nose_convlayers = [
            conv(3, 32, 3, 2, 1), nn.LeakyReLU(eta),
            conv(32, 64, 3, 2, 1), nn.LayerNorm([11, 8]), nn.LeakyReLU(eta),
            conv(64, 128, 3, 2, 1), nn.LayerNorm([6, 4]), nn.LeakyReLU(eta),
            conv(128, 256, 3, 2, 1), nn.LayerNorm([3, 2]), nn.LeakyReLU(eta),
            Flatten(), nn.Linear(1536, 1)
        ]
        mouth_convlayers = [
            conv(3, 32, 3, 2, 1), nn.LeakyReLU(eta),
            conv(32, 64, 3, 2, 1), nn.LayerNorm([6, 11]), nn.LeakyReLU(eta),
            conv(64, 128, 3, 2, 1), nn.LayerNorm([3, 6]), nn.LeakyReLU(eta),
            conv(128, 256, 3, 2, 1), nn.LayerNorm([2, 3]), nn.LeakyReLU(eta),
            Flatten(), nn.Linear(1536, 1)
        ]
        face_convlayers = [
            conv(3, 32, 3, 2, 1), nn.LeakyReLU(eta),
            conv(32, 64, 3, 2, 1), nn.LayerNorm([22, 23]), nn.LeakyReLU(eta),
            conv(64, 128, 3, 2, 1), nn.LayerNorm([11, 12]), nn.LeakyReLU(eta),
            conv(128, 256, 3, 2, 1), nn.LayerNorm([6, 6]), nn.LeakyReLU(eta),
            Flatten(), nn.Linear(9216, 1)
        ]
        self.eyes_convLayers = nn.Sequential(*eyes_convlayers)
        self.nose_convLayers = nn.Sequential(*nose_convlayers)
        self.mouth_convLayers = nn.Sequential(*mouth_convlayers)
        self.face_convLayers = nn.Sequential(*face_convlayers)
        self.image_connLayers = nn.Sequential(*img_convlayers)

    def forward(self, x):
        return self.Disc(x)

    def Disc(self, img):
        eyes_ROI, nose_ROI, mouth_ROI, face_ROI = segment(img)
        eys_score = self.eyes_convLayers(eyes_ROI)
        nose_score = self.nose_convLayers(nose_ROI)
        mouth_score = self.mouth_convLayers(mouth_ROI)
        face_score = self.face_convLayers(face_ROI)
        img_score = self.image_connLayers(img)

        return img_score, eys_score, nose_score, mouth_score, face_score

    @staticmethod
    def get_grad(output, interpolates):
        grad_of_local = grad(outputs=output.sum(),
                             inputs=interpolates,
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]
        return grad_of_local

    def CriticWithGP_Loss(self, fake_gan, real_gan, interpolates):
        Syn_F = fake_gan[0] + fake_gan[1] + fake_gan[2] + fake_gan[3] + fake_gan[4]
        Real = real_gan[0] + real_gan[1] + real_gan[2] + real_gan[3] + real_gan[4]
        wasserstein_dis = (Syn_F-Real).mean() / 5
        inter = self.Disc(interpolates)
        grad_of_eye = self.get_grad(inter[1], interpolates)
        grad_of_nose = self.get_grad(inter[2], interpolates)
        grad_of_mouth = self.get_grad(inter[3], interpolates)
        grad_of_face = self.get_grad(inter[4], interpolates)
        grad_of_img = self.get_grad(inter[0], interpolates)
        gradients = grad_of_eye+grad_of_nose+grad_of_mouth+grad_of_face+grad_of_img
        # gradient penalty
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=-1)-1) ** 2).mean()
        loss = wasserstein_dis + 10 * gradient_penalty

        return loss


if __name__ == '__main__':
    t = torch.randn(2, 3, 128, 128)
    m1 = torch.randn(2, 1, 128, 128)
    m2 = torch.randn(2, 1, 128, 128)
    G = Generator()
    # Da = Discriminator_Attention()
    # Dm = Discriminator_mark()
    # local = Da(t)
    # print(local[0])
    # out = Dm(torch.cat([t, m1], dim=1))
    # print(out)
    o1, o2, o3 = G(t, m1, m2)
    print(o1.shape, o2.shape, o3.shape)


