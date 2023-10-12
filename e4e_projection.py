import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from e4e.models.psp import pSp
from util import *


@ torch.no_grad()
def projection(img, name, generator=None, device='cuda'):
    model_path = '/home/ishant/Desktop/test2/RetrieveInStyle/e4e_ffhq_encode.pt'
    # ensure_checkpoint_exists(model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    filename = './inversion_codes/' + name + '.pt'
    result_file['latent'] = w_plus[0]
    torch.save(result_file, filename)
    image_np = images.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Assuming your tensor is in the format CxHxW

    # Convert to a PIL Image
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Save the image
    image_pil.save("kk1.jpg")

face = Image.open('resized_image.jpg')
if face.mode == 'RGBA':
    face = face.convert('RGB')

projection(face , "test5" )
