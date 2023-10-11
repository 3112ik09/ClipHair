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
from tqdm import tqdm
sys.path.append(".")
sys.path.append("..")
from utils.common import tensor2im

from models.e4e_features2 import pSp
# from models.psp_features2 import pSp
from mapper.hairclip_mapper2 import HairCLIPMapper
from mapper.hairclip_mapper_gn import HairCLIPMapper as HairCLIPMapper_gn

def get_avg_image(encoder, mapper, face_pool):
    avg_image, _ = mapper.decoder([encoder.latent_avg.unsqueeze(0)],
                                            input_is_latent=True,
                                            randomize_noise=False,
                                            return_latents=False)
    
    avg_image = face_pool(avg_image)
    avg_image = avg_image.to('cuda').squeeze(0).float().detach()
    return avg_image

captions_path = "/scratch/users/abaykal20/sam/SAM/mmcelebhq/test_captions/"
captions_dir = sorted(os.listdir(captions_path))
images_path = "/scratch/users/abaykal20/sam/SAM/mmcelebhq/test_images/"
images_dir = sorted(os.listdir(images_path))


EXPERIMENT_DATA_ARGS = {
    "celeba_encode": {
        "model_path": "exp_text_cycle_lighterer_cont/checkpoints/best_model.pt",
        "e4e_path": "/scratch/users/abaykal20/sam/SAM/pretrained_models/e4e_ffhq_encode.pt",
        "transform": transforms.Compose([
            # transforms.CenterCrop((178,178)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    
    "celeba_encode_gn": {
        "model_path": "exp_groupnorm/checkpoints/best_model.pt",
        "e4e_path": "/scratch/users/abaykal20/sam/SAM/pretrained_models/e4e_ffhq_encode.pt",
        "transform": transforms.Compose([
            # transforms.CenterCrop((178,178)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS["celeba_encode"]

EXPERIMENT_ARGS_gn = EXPERIMENT_DATA_ARGS["celeba_encode_gn"]

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

model_path_gn = EXPERIMENT_ARGS_gn['model_path']
ckpt_gn = torch.load(model_path_gn, map_location='cpu')
opts_gn = ckpt_gn['opts']
opts_gn['checkpoint_path'] = model_path_gn
opts_gn = Namespace(**opts_gn)
encoder_gn = pSp(opts_gn)
encoder_gn.eval()
encoder_gn.cuda()

mapper_gn = HairCLIPMapper_gn(opts_gn)
mapper_gn.eval()
mapper_gn.cuda()
print("Models Succesfully Loaded!")

clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
print("Starting inference")


wf = open("gn_comparison/generated_captions.txt", "w")
wi = open("gn_comparison/image_indexes.txt", "w")

img_transforms = EXPERIMENT_ARGS['transform']
for i in range(30):
    image_idx = np.random.randint(0,6000)
    complete_image_path = os.path.join(images_path, images_dir[image_idx])
    caption_idx = np.random.randint(0,6000)
    complete_caption_path = os.path.join(captions_path, captions_dir[caption_idx])
    original_image = Image.open(complete_image_path).convert("RGB")
    f = open(complete_caption_path, "r")
    caption = f.readline()
    f.close()
    wf.write(caption)
    wi.write(images_dir[image_idx])
    wi.write("\n")
    img_transforms = EXPERIMENT_ARGS['transform']
    input_image = img_transforms(original_image)
    input_image = input_image.unsqueeze(0)
    text_input = clip.tokenize(caption)
    text_input = text_input.cuda()
    input_image = input_image.cuda().float()
    results = np.array(original_image)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input).float()
        
        w, features = encoder.forward(input_image, return_latents=True)
        features = mapper.mapper(features, text_features)
        w_hat = w + 0.1 * encoder.forward_features(features)

        result_tensor, _ = mapper.decoder([w_hat], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1)
        result_tensor = result_tensor.squeeze(0)
        result_image = tensor2im(result_tensor)
        
        w_gn, features_gn = encoder_gn.forward(input_image, return_latents=True)
        features_gn = mapper_gn.mapper(features_gn, text_features)
        w_hat_gn = w_gn + 0.1 * encoder_gn.forward_features(features_gn)
        
        result_tensor_gn, _ = mapper_gn.decoder([w_hat_gn], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1)
        result_tensor_gn = result_tensor_gn.squeeze(0)
        result_image_gn = tensor2im(result_tensor_gn)
        
        results = np.concatenate([results, result_image, result_image_gn], axis=1)
        results = Image.fromarray(results)
        results.save(f"gn_comparison/{i}.jpg")

wf.close()
wi.close()
print("Finished inference")