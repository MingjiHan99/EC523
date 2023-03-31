from data import Dataset
from pretrained_cnn import PretrainedModel
from matplotlib import pyplot as plt
import torchvision.transforms as T
import numpy as np
import torch 
from pd_gan import PDGANGenerator, PDGANDiscriminator
from PIL import Image
from pretrained_loss import GANLoss, Diversityloss, PerceptualLoss
import cv2
import tqdm

def save_img(tensor, name):
    npimg = np.transpose(tensor, (1,2,0)) + 0.5
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
    
# Img and mask are in pytorch tensor format
def test_gan_model(generator, pretrained_cnn, imgs, masks):
    result = pretrained_cnn([imgs, masks])
    z = torch.randn((imgs.shape[0], 256)).cuda()
    fake = generator(z, result, masks)
    b  = fake.shape[0]
    for i in range(b):
        img = fake[i].cpu().detach().numpy()
        save_img(img, '{}.png'.format(i))
   
if __name__ == "__main__":
    # Define dataset
    dataset = Dataset('./data/celeba/', True, './data/mask/testing_mask_dataset/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    # Define pretrained model
    pretrained_cnn = PretrainedModel('./Face/2_net_EN.pth', './Face/2_net_DE.pth')
    # Define PDGAN
    generator = PDGANGenerator()
    generator = generator.cuda()
    discriminator = PDGANDiscriminator()
    discriminator = discriminator.cuda()
    # Losses
    gan_loss = GANLoss()
    preceptual_loss = PerceptualLoss()
    preceptual_divsersity_loss = Diversityloss()
    # Training Parameters
    epoch = 20
    lr = 0.001
    # Optimizer
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.999))
    
    
    for i in range(epoch):
        for imgs, masks in tqdm(dataloader):
            # Get the input images and their corresponding masks
            imgs = imgs.cuda()
            masks = masks.cuda()
            masks = masks.repeat(1, 3, 1, 1)
            # Backup the original images and masks
            original_imgs = imgs.detach().clone()
            hole_mask = masks.detach().clone()
            
            # Get the masked images
            masks = 1 - masks
            imgs = imgs * masks
            
            # Get the preliminary fixed images from pretrained model
            raw_fix_imgs = pretrained_cnn.forward([imgs, masks])
            # Train GAN
            # Generate fake images
            z0 = torch.randn((imgs.shape[0], 256)).cuda()
            z1 = torch.randn((imgs.shape[0], 256)).cuda()
            fake0 = generator(z0, raw_fix_imgs, masks)
            fake1 = generator(z1, raw_fix_imgs, masks)
            
            # Train discriminator
            
            # Fix generator and clean gradients
            for p in generator.parameters():
                p.requires_grad = False
            for p in discriminator.parameters():
                p.requires_grad = True
            discriminator_optimizer.zero_grad()
            # Compute loss
            
            fake0_dis_input = torch.cat([hole_mask, fake0], dim=1)
            fake1_dis_input = torch.cat([hole_mask, fake1], dim=1)
            real_dis_input = torch.cat([hole_mask, imgs], dim=1)
            
            pred_fake_0 = discriminator(fake0_dis_input)
            pred_fake_1 = discriminator(fake0_dis_input)
            pred_real = discriminator(real_dis_input)
            loss_d = gan_loss(pred_fake_0, False, for_discriminator=True) + gan_loss(pred_fake_1, False, for_discriminator=True) + gan_loss(pred_real, True, for_discriminator=True)
            
            loss_d.backward()
            discriminator_optimizer.step()
            
            # Train generator
            # Fix discriminator and clean gradients
            for p in generator.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False
            generator_optimizer.zero_grad()
            # Compute loss
            
            # GAN Loss
            loss_g = gan_loss(pred_fake_0, False, for_discriminator=False) + gan_loss(pred_fake_1, False, for_discriminator=False) + gan_loss(pred_real, True, for_discriminator=False)
            
            # Perceptual Loss
            loss_g = loss_g + preceptual_loss(fake0, fake1)
            
            # Perceptual Diversity Loss
            loss_g = loss_g + preceptual_divsersity_loss(fake0, fake1)
            
            loss_g.backward()
            generator_optimizer.step()
            
            
        # Store model and test model
        torch.save(generator.state_dict(), './model/generator.pth')
        torch.save(discriminator.state_dict(), './model/discriminator.pth')
        