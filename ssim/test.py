import torch
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim, ms_ssim

#img2 = ToTensor()(Image.open('test_0.png'))

def process():
    # Load the images
    img1 = ToTensor()(Image.open('input_4.png'))
    img2 = ToTensor()(Image.open('test_4_trial_2.png'))
    # Calculate SSIM and MS-SSIM
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif len(img1.shape) != 4:
        raise ValueError("Input images should have 3 or 4 dimensions.")
    if img1.shape != img2.shape:
        raise ValueError("Input images should have the same dimensions.")
    ssim_val = ssim(img1, img2, data_range=1.0, size_average=False) # return (N,)
    ms_ssim_val = ms_ssim(img1, img2, data_range=1.0, size_average=False) # return (N,)
    print(ssim_val)
    print(ms_ssim_val)
    
process()
