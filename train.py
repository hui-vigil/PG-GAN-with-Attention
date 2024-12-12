from tools import utils
from config import *


def train():
    step = 0
    G.train()
    D_attention.train()
    D_landmark.train()
    for epoch in range(args.start_epoch+1, args.epochs+1):

        for index, data in enumerate(train_loader):
            for key in data.keys():
                data[key] = data[key].to(args.device)
            img32, img64, img128 = G(data['profile'], data['real_mark'], data['front_mark'])

            # update D_Attention
            utils.set_requires_grad(D_attention, True)
            syn_front = D_attention(img128.detach())
            real_front = D_attention(data['front'])
            gp_alpha = torch.rand(data['front'].size(0), 1, 1, 1)
            gp_alpha = gp_alpha.expand_as(img128).clone().pin_memory().to(args.device)
            t1 = torch.ones_like(gp_alpha)
            interpolated = gp_alpha * img128.detach() + (t1 - gp_alpha) * data['front']
            interpolated = interpolated.to(args.device).requires_grad_()
            D_A_Loss= D_attention.CriticWithGP_Loss(syn_front, real_front, interpolated)
            optimizer_D_attention.zero_grad()
            D_A_Loss.backward()
            optimizer_D_attention.step()

            # update D_landmark
            utils.set_requires_grad(D_attention, False)
            utils.set_requires_grad(D_landmark, True)

            D_lm_real_input = torch.cat([data['front'], data['front_mark']], dim=1)
            D_lm_fake_input = torch.cat([img128.detach(), data['front_mark']], dim=1)
            D_real_output = D_landmark(D_lm_real_input)
            D_fake_output = D_landmark(D_lm_fake_input)
            gp_alpha = torch.rand(data['front'].size(0), 1, 1, 1)
            gp_alpha = gp_alpha.expand_as(D_lm_real_input).clone().pin_memory().to(args.device)
            t2 = torch.ones_like(gp_alpha)
            interpolated_2 = gp_alpha * D_lm_fake_input + (t2 - gp_alpha) * D_lm_real_input
            interpolated_2 = interpolated_2.to(args.device).requires_grad_()
            D_lm_Loss = utils.get_gp_loss(D_real_output, D_fake_output, interpolated_2, D_landmark, args.device)

            optimizer_D_landmark.zero_grad()
            D_lm_Loss.backward()
            optimizer_D_landmark.step()

            utils.set_requires_grad(D_landmark, False)
            # update G
            utils.set_requires_grad(G, True)

            adv1 = G.GLoss(D_attention(img128))
            # adv1 = -torch.mean(D_attention(torch.cat([img120, data['profile']], dim=1)))
            adv2 = -torch.mean(D_landmark(torch.cat([img128, data['front_mark']], dim=1)))

            # 逐像素损失
            pixel_128_loss = L1(img128, data['front'])
            pixel_64_loss = L1(img64, data['img_64'])
            pixel_32_loss = L1(img32, data['img_32'])
            # pixel_loss = (pixel_30_loss + pixel_60_loss + pixel_120_loss) / 3

            # 身份感知损失，余弦距离衡量
            features_real = extract_net(data['front'])  # fetures_real
            features_fake = extract_net(img128)  # fetures_fake
            # cos_sim = cosine(features_fake, features_real)
            # id_loss = torch.mean(1.0 - cos_sim) #
            id_loss = mse(features_real, features_fake)

            # 全变分损失
            total_var = total_Var(img128)

            total_G_loss = 10 * pixel_128_loss + 0.1 * pixel_64_loss + (
                1e-4) * pixel_32_loss + args.lambda_adv1 * adv1 + \
                           args.lambda_adv2 * adv2 + args.lambda_id * id_loss + args.lambda_tv * total_var

            optimizer_G.zero_grad()
            total_G_loss.backward()
            optimizer_G.step()

        if epoch % 4 == 0:
            utils.sample_images(G, train_loader, val_loader, epoch, args.device, args.imgout)
            utils.save_model({
                'epoch': epoch,
                'G_param': G.state_dict(),
                'D_landmark_param': D_landmark.state_dict(),
                'D_attention_param': D_attention.state_dict()
            }, args.modelout, epoch)



