from data import Dataset
from pretrained_cnn import PretrainedModel
from matplotlib import pyplot as plt
import numpy as np
import torch 
from pd_gan import PDGANGenerator, PDGANDiscriminator
from pretrained_loss import GANLoss, Diversityloss, PerceptualLoss, TVloss
from tqdm import tqdm
from torch.nn import init
def save_img(tensor, name):
    npimg = (np.transpose(tensor, (1,2,0)) + 1.0) / 2.0 * 255
    npimg =  np.minimum(np.maximum(npimg, 0), 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.savefig(name)
    
def save_img_tanh(tensor, name):
    npimg = (np.transpose(tensor, (1,2,0)) + 1.0)  / 2.0 * 255
    npimg =  np.minimum(np.maximum(npimg, 0), 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.savefig(name)

def test_pre_train_model(dataset):
    img, mask = dataset[999]
    img = img.cuda()
    mask = mask.cuda()
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    if mask.shape[1] == 1:
        mask = mask.repeat(1, 3, 1, 1)
    mask = 1 - mask
    img = img * mask
    print(img.shape)
    print(mask.shape) 
   
    pretrained_cnn = PretrainedModel('./Face/2_net_EN.pth', './Face/2_net_DE.pth')
    result = pretrained_cnn.forward([img, mask])
    result = result.squeeze(0)
    
    result = result.cpu().numpy()
    
    save_img(result, 'result.png')
    
def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
        init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, "weight") and "BatchNorm2d" in classname:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
# Img and mask are in pytorch tensor format
def test_gan_model(generator, pretrained_cnn, imgs, masks, epoch):
    result = pretrained_cnn.forward([imgs, masks])
    z = torch.randn((imgs.shape[0], 256)).cuda()
    with torch.no_grad():
        fake = generator(z, result, masks)
        b  = fake.shape[0]
        for i in range(b):
          #  
         #   save_img(result[i].cpu().detach().numpy(), './log/input_{}_{}.png'.format(epoch, i))
            img = fake[i].cpu().detach().numpy() 
            concat = fake[i] * masks[i] + imgs[i] * (1 - masks[i])
            save_img_tanh(img, './log/{}_{}.png'.format(epoch, i))
   
if __name__ == "__main__":
    # Define dataset
    dataset_path = './data/celeba_small/'
    mask_path = './data/mask/testing_mask_dataset/'
    encoder_path = './Face/2_net_EN.pth'
    decoder_path = './Face/2_net_DE.pth'
    dataset = Dataset(dataset_path, True, mask_path)
    img_samples = []
    mask_samples = []
    for i in range(4):
        img, mask = dataset[i + 250]
        img_samples.append(img.unsqueeze(0))
        mask_samples.append(mask.unsqueeze(0))
    
    img_samples = torch.cat(img_samples, dim = 0)
    mask_samples = torch.cat(mask_samples, dim = 0)

    mask_samples = 1 - mask_samples
    mask_samples = mask_samples.repeat(1, 3, 1, 1)
    img_samples = img_samples * mask_samples
    for i in range(4):
        save_img(img_samples[i].numpy(), './log/input_{}.png'.format(i))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=12)
    # Define pretrained model
    pretrained_cnn = PretrainedModel('./Face/2_net_EN.pth', './Face/2_net_DE.pth')
    # Define PDGAN
    generator = PDGANGenerator()
    generator = generator.cuda()
    generator.apply(init_func)
    discriminator = PDGANDiscriminator()
    discriminator = discriminator.cuda()
    discriminator.apply(init_func)
    # Losses
    gan_loss = GANLoss('hinge')
    gan_loss = gan_loss.cuda()
    preceptual_loss = PerceptualLoss()
    preceptual_loss = preceptual_loss.cuda()
    preceptual_divsersity_loss = Diversityloss()
    preceptual_divsersity_loss = preceptual_divsersity_loss.cuda()
    # Training Parameters
    epochs = 100
    lr = 0.001
    current_lr = lr
    # Optimizer
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
    l1_loss = torch.nn.L1Loss()
    for epoch in range(epochs):
        bar = tqdm(dataloader)
        for imgs, masks in bar:
            # Get the input images and their corresponding masks
            imgs = imgs.cuda()
            masks = masks.cuda()
            masks = masks.repeat(1, 3, 1, 1)
            # Backup the original images and masks
            original_imgs = imgs.detach()
            hole_mask = masks.detach()
            
            # Get the masked images
            masks = 1 - masks
            imgs = imgs * masks
            
            # Get the preliminary fixed images from pretrained model
            raw_fix_imgs = pretrained_cnn.forward([imgs, masks])
            # Train GAN
            raw_fix_imgs = raw_fix_imgs.detach()
            z0 = torch.randn((imgs.shape[0], 256)).cuda()
            z1 = torch.randn((imgs.shape[0], 256)).cuda()
            fake0 = generator(z0, raw_fix_imgs, masks)
            fake1 = generator(z1, raw_fix_imgs, masks)
            
            
            # Fix generator and clean gradients
            for p in generator.parameters():
                p.requires_grad = False
            for p in discriminator.parameters():
                p.requires_grad = True
            
            discriminator_optimizer.zero_grad()
            dis_fake0 = fake0.detach()
            dis_fake1 = fake1.detach()
            dis_fake0_input = torch.cat([hole_mask, dis_fake0], dim=1)
            dis_fake1_input = torch.cat([hole_mask, dis_fake1], dim=1)
            real_dis_input = torch.cat([hole_mask, original_imgs], dim=1)
            
            pred_fake_0 = discriminator(dis_fake0_input)
            pred_fake_1 = discriminator(dis_fake1_input)
            pred_real = discriminator(real_dis_input)
            # Compute loss
            loss_d = gan_loss(pred_fake_0, False, for_discriminator=True) + gan_loss(pred_fake_1, False, for_discriminator=True) + 2.0 * gan_loss(pred_real, True, for_discriminator=True)
            
            loss_d.backward(retain_graph=True)
            discriminator_optimizer.step()
            
            # Train generator
            # Fix discriminator and clean gradients
            for p in generator.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False
            generator_optimizer.zero_grad()
            gen_fake0_input = torch.cat([hole_mask, fake0], dim=1)
            gen_fake1_input = torch.cat([hole_mask, fake1], dim=1)
            gen_pred_fake_0 = discriminator(gen_fake0_input)
            gen_pred_fake_1 = discriminator(gen_fake1_input)
            
            num_of_output = len(gen_pred_fake_0)
            feature_matching_loss = torch.zeros((1, )).cuda()
            for i in range(num_of_output):  
                num_intermediate_outputs = len(gen_pred_fake_0[i]) - 1
                for j in range(num_intermediate_outputs): 
                    single_mactching_loss = l1_loss(
                        gen_pred_fake_0[i][j], pred_real[i][j].detach()
                    ) + l1_loss(
                        gen_pred_fake_1[i][j], pred_real[i][j].detach()
                    )
                    feature_matching_loss += single_mactching_loss * 10.0 / num_of_output
            # GAN Loss
            loss_g = feature_matching_loss
            loss_g = gan_loss(gen_pred_fake_0, target_is_real=True, for_discriminator=False) \
                    + gan_loss(gen_pred_fake_1, target_is_real=True, for_discriminator=False)
            # Perceptual Loss
            loss_g = loss_g + (10.0 * preceptual_loss(fake0, original_imgs) \
                            + 10.0 * preceptual_loss(fake1, original_imgs)) 
            # Perceptual Diversity Loss
            loss_g = loss_g + 1.0 / (preceptual_divsersity_loss(fake0 * hole_mask, fake1 * hole_mask) + 1 * 1e-5)
            
            # TV Loss
            comp_0 = hole_mask * fake0 + masks * original_imgs
            comp_1 = hole_mask * fake1 + masks * original_imgs
            loss_g += TVloss(comp_0, masks, 'mean') + TVloss(comp_1, masks, 'mean')
            
            loss_g = loss_g / 5.0
            loss_g.backward()
            generator_optimizer.step()
            bar.set_postfix(G_Loss=loss_g.item(), D_Loss=loss_d.item())
        # Store model and test model
        torch.save(generator.state_dict(), './model/generator.pth')
        torch.save(discriminator.state_dict(), './model/discriminator.pth')
        test_gan_model(generator, pretrained_cnn, img_samples.cuda(), mask_samples.cuda(), epoch)