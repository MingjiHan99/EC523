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
    npimg = (np.transpose(tensor, (1,2,0)) + 1.0)  / 2.0 * 255
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
    
# Img and mask are in pytorch tensor format
def test_gan_model(generator, pretrained_cnn, imgs, masks, ):
    result = pretrained_cnn.forward([imgs, masks])
    # Try to generate 5 images
    for k in range(10):
        z = torch.randn((imgs.shape[0], 256)).cuda()
        with torch.no_grad():
            fake = generator(z, result, masks)
            b  = fake.shape[0]
            for i in range(b):
                img = fake[i].cpu().detach().numpy() 
                save_img_tanh(img, './log/test_{}_trial_{}.png'.format(i, k))
if __name__ == "__main__":
     # Define dataset
    dataset_path = './data/celeba_small/'
    mask_path = './data/mask/testing_mask_dataset/'
    encoder_path = './Face/2_net_EN.pth'
    decoder_path = './Face/2_net_DE.pth'
    dataset = Dataset('./data/celeba_small/', True, './data/mask/testing_mask_dataset/')
    img_samples = []
    mask_samples = []
    for i in range(5):
        img, mask = dataset[i + 23]
        img_samples.append(img.unsqueeze(0))
        mask_samples.append(mask.unsqueeze(0))
        
    img_samples = torch.cat(img_samples, dim = 0)
    mask_samples = torch.cat(mask_samples, dim = 0)

    mask_samples = 1 - mask_samples
    mask_samples = mask_samples.repeat(1, 3, 1, 1)
    img_samples = img_samples * mask_samples
    for i in range(5):
        save_img(img_samples[i].numpy(), './log/input_{}.png'.format(i))
    
    ataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=12)
    # Define pretrained model
    pretrained_cnn = PretrainedModel('./Face/2_net_EN.pth', './Face/2_net_DE.pth')
    # Define PDGAN
    generator = PDGANGenerator()
    model_kv = torch.load('./model_250/generator.pth')
    generator.load_state_dict(model_kv)
    generator = generator.cuda()
    test_gan_model(generator, pretrained_cnn, img_samples.cuda(), mask_samples.cuda())