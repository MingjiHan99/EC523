import numpy
import math
import imageio

pth1 = "./log/input_1.png"
pth2 = "./log/test_1_trial_0.png"

img1 = imageio.imread(pth1)
img2 = imageio.imread(pth2)

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
