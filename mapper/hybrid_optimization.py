import os
from argparse import Namespace
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
from tqdm import tqdm
import sys
import torch.nn as nn
import math

sys.path.append(".")
sys.path.append("..")
from utils.common import tensor2im
from criteria.lpips.lpips import LPIPS
from criteria import id_loss, w_norm
from criteria.clip_loss import CLIPLoss
from models.perceptual_model import PerceptualModel

from models.e4e_features2 import pSp
from mapper.hairclip_mapper2 import HairCLIPMapper

def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp
# captions_path = "mmcelebhq/test_captions/"
# captions_dir = sorted(os.listdir(captions_path))
# images_path = "mmcelebhq/test_images/"
# images_dir = sorted(os.listdir(images_path))

# EXPERIMENT_TYPE = 'celeba_encode'
EXPERIMENT_DATA_ARGS = {
    "celeba_encode": {
        "model_path": "exp_text_cycle_save/checkpoints/best_model.pt",
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
print("Models Succesfully Loaded!")

clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')

# wf = open("hybrid_out/comparison_tedigan/generated_captions.txt", "w")
# wi = open("hybrid_out/comparison_tedigan/image_indexes.txt", "w")

# loss_pix_weight = 1.0
loss_pix_weight = 0.0
# loss_feat_weight = 5e-5
loss_feat_weight = 0
# loss_reg_weight = 4.0
loss_reg_weight = 0.004
loss_clip_weight = 1.0
clip_loss = CLIPLoss(clip_model)
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
F = PerceptualModel()
id_loss = id_loss.IDLoss().cuda().eval()
# id_loss_weight = 0.1
# id_loss_weight = 0.2
id_loss_weight = 0.005

# for normal edits pix = 0.008, feat = 5e-5, reg = 1.0, clip = 2.0, id = 0.2
# for id edits pix = 0.008, feat = 5e-5, reg = 0.2, clip = 3.0, id = 0

mse_loss = nn.MSELoss().cuda().eval()
lpips_loss = LPIPS(net_type='alex').cuda().eval()
w_norm_loss = w_norm.WNormLoss(opts=opts)

print("Starting inference")
img_transforms = EXPERIMENT_ARGS['transform']
# captions = ["Wrinkle",
#             "Sad",
#             "Angry",
#             "Surprised",
#             "Beard",
#             "Bald",
#             "Grey Hair",
#             "Black Hair"]

captions = ["Mohawk hairstyle",
           "Curly hair",
           "Bob-cut hairstyle",
           "Afro hairstyle"]

for q in range(4):

    complete_image_path = "hybrid_comparisons/fig4/inversion/00111.jpg"
    # custom_caption = "Wrinkle"
    custom_caption = captions[q]
    original_image = Image.open(complete_image_path).convert("RGB")
    input_image = img_transforms(original_image)
    input_image = input_image.unsqueeze(0).cuda().float()
    text_input = clip.tokenize(custom_caption)
    text_input = text_input.cuda()
    results = np.array(original_image)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input).float()

        w, features = encoder.forward(input_image, return_latents=True)
        features = mapper.mapper(features, text_features)
        residual_init = encoder.forward_features(features)
        result_tensor, _ = mapper.decoder([w + 0.1 * residual_init], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1)
        encoder_out_tensor, _ = mapper.decoder([w], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1)
        result_tensor = result_tensor.squeeze(0)
        result_image = tensor2im(result_tensor)

    residual = residual_init.detach().clone()
    residual.requires_grad = True

    optimizer = torch.optim.Adam([residual], lr=0.1)
    num_iterations = 300
    pbar = tqdm(range(1, num_iterations + 1), leave=True)
    x = result_tensor.unsqueeze(0)
    x = face_pool(x)
    enc_out = face_pool(encoder_out_tensor)
    input_image_resized = face_pool(input_image)

    print(f'Starting latent optimization for caption: {custom_caption}')
    for step in pbar:
        t = step / num_iterations
        lr = get_lr(t, 0.1)
        optimizer.param_groups[0]["lr"] = lr

        loss = 0.0
        # Reconstruction loss.
        x_rec, _ = mapper.decoder([w + 0.1 * residual], input_is_latent=True, randomize_noise=False, return_latents=False)
        x_rec = face_pool(x_rec)
        # loss_pix = torch.mean((x - x_rec) ** 2)
        loss_pix = mse_loss(x_rec , x)
        loss = loss + loss_pix * loss_pix_weight
        log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

        # Perceptual loss.
        if loss_feat_weight:
        # x_feat = F.net(x)
        # x_rec_feat = F.net(x_rec)
        # loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
            loss_feat = lpips_loss(x_rec, x)
            loss = loss + loss_feat * loss_feat_weight
            log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

        # Regularization loss.
        if loss_reg_weight:
            loss_reg = torch.sum((0.1 * (residual )) ** 2) # - residual_init
            loss = loss + loss_reg * loss_reg_weight
            log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'

        # CLIP loss.
        if loss_clip_weight:
            loss_clip = clip_loss(x_rec, text_input)
            loss = loss + loss_clip[0][0] * loss_clip_weight
            log_message += f', loss_clip: {_get_tensor_value(loss_clip[0][0]):.3f}'

        # ID loss.
        if id_loss_weight:
            loss_id, _, _ = id_loss(x_rec, input_image_resized, input_image_resized)
            loss = loss + loss_id * id_loss_weight
            log_message += f', loss_id: {_get_tensor_value(loss_id):.3f}'

        log_message += f', loss: {_get_tensor_value(loss):.3f}'
        pbar.set_description_str(log_message)

        # Do optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    optimized_res, _ = mapper.decoder([w + 0.1 * residual], input_is_latent=True, randomize_noise=False, return_latents=False)
    optimized_res = optimized_res.squeeze(0)
    styleclip = Image.open(f"hybrid_comparisons/fig4/out/{q+1}.jpg").convert("RGB")

    optimized_res_img = tensor2im(optimized_res)
    results = np.concatenate([results, styleclip, result_image, optimized_res_img], axis=1)
    results = Image.fromarray(results)
    results.save(f"hybrid_comparisons/fig4/out/out_{q+1}.jpg")
 
print("Finished inference")