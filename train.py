from data import Dataset
from pretrained_cnn import PretrainedModel
from matplotlib import pyplot as plt
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

dataset = Dataset('./data/celeba/', True, './data/mask/testing_mask_dataset/')
img, mask = dataset[999]
img = img * (1 - mask)
img = img.numpy()

npimg = np.transpose(img,(1,2,0)) + 0.5
plt.imshow(npimg)

plt.savefig('out.png')