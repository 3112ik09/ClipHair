import os
from argparse import Namespace
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
import sys
sys.path.append(".")
sys.path.append("..")
from utils.common import tensor2im

from models.e4e import pSp
from mapper.hairclip_mapper_text import HairCLIPMapper

captions_dir = "/scratch/users/abaykal20/sam/SAM/mmcelebhq/inference_captions.txt"
f = open(captions_dir, "r")
all_captions = f.readlines()
images_path = "/scratch/users/abaykal20/sam/SAM/mmcelebhq/test_images/"
images_dir = sorted(os.listdir(images_path))

EXPERIMENT_TYPE = 'celeba_encode'
EXPERIMENT_DATA_ARGS = {
    "celeba_encode": {
        "model_path": "exp_text_norm2/checkpoints/best_model.pt",
        "e4e_path": "/scratch/users/abaykal20/sam/SAM/pretrained_models/e4e_ffhq_encode.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS["celeba_encode"]

print("Loading Models")
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
encoder = pSp(opts)
encoder.eval()
encoder.cuda()

mapper = HairCLIPMapper(opts)
mapper.eval()
mapper.cuda()
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
print("Models Succesfully Loaded!")

clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')

print("Starting inference")
for idx, img_path in enumerate(images_dir):
    complete_image_path = os.path.join(images_path, img_path)
    original_image = Image.open(complete_image_path).convert("RGB")
    # original_image.resize((256, 256))
    custom_caption = all_captions[idx].rstrip()
    img_transforms = EXPERIMENT_ARGS['transform']
    input_image = img_transforms(original_image)
    input_image = input_image.unsqueeze(0)
    text_input = clip.tokenize(custom_caption)
    text_input = text_input.cuda()
    input_image = input_image.cuda().float()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input).float()
        # features
        # w, features = encoder.forward(input_image, return_latents=True)
        # features = mapper.mapper(features, text_features)
        # w_hat = w + 0.1 * encoder.forward_features(features)
        
        # no multiplier
        # w, features = encoder.forward(input_image, return_latents=True)
        # features = mapper.mapper(features, text_features)
        # w_hat = w + encoder.forward_features(features)
        
        # # modified hairclip
        w = encoder.forward(input_image, return_latents=True)
        w_hat = w + 0.1 * mapper.mapper(w, text_features)
        result_tensor, _ = mapper.decoder([w_hat], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1)
        result_tensor = face_pool(result_tensor)
        result_tensor = result_tensor.squeeze(0)
        result_image = tensor2im(result_tensor)
        
        # results = np.concatenate([results, result_image], axis=1)
        # results = Image.fromarray(results)
        result_image.save("fid_images/text/{}.png".format(idx))
        
print("Finished inference")