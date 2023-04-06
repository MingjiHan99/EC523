### PD-GAN

EC523 Project

Image Impainting using PD-GAN (CVPR 2021)

# Training:

1. Download the image dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

NOTICE: If the training time is too long, you can randomly choose some images and put them in a photo.

2. Download the mask dataset: https://nv-adlr.github.io/publication/partialconv-inpainting

2. Download the pretrained encoder and decoder: https://drive.google.com/drive/folders/1o9reT5_lFzGKBsrLlvck545nNInIAlPe

3. Create folder for training log and model: 

```
mkdir model
mkdir log
```
4. modify the path in `train.py`
```
dataset_path = 'your own path'
mask_path = 'your own path'
encoder_path = 'your own path'
decoder_path = 'your own path'
```

5. Start training:

```
python3 train.py
```

# Test:

```
python3 test.py
```

### Acknowledgement

The code and model of Pretrained Encoder-Decoder for building PD-GAN are adapted from the following sources:

https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/pconv.py
