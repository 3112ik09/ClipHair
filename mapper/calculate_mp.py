import torch.nn as nn
from torchvision import transforms
import torch
import numpy as np
import os
import clip
from PIL import Image
from tqdm import tqdm

captions_dict = {
                 "Blond_Hair" : "This person has blond hair.",
                 "Bushy_Eyebrows" : "This person has bushy eyebrows.",
                 "Chubby" : "This person is chubby.",
                 "Double_Chin" : "This person has double chin.",
                 "Eyeglasses" : "This person has eyeglasses.",
                 "Goatee" : "This person has a goatee.",
                 "Gray_Hair" : "This person has gray hair.",
                 "Heavy_Makeup" : "This person wears heavy makeup.",
                 "Male" : "This is a male.",
                 "Mouth_Slightly_Open" : "This person has mouth slightly open.",
                 "Mustache" : "This person has a mustache.",
                 "Rosy_Cheeks" : "This person has rosy cheeks.",
                 "Smiling" : "This person is smiling.",
                 "Wearing_Lipstick" : "This person is wearing lipstick.",
                 "Wearing_Necktie" : "This person is wearing a necktie."}

# image_folder = "/scratch/users/abaykal20/sam/SAM/attribute_classification/"
image_folder = "attribute_classification/"
# image_folder = "/scratch/users/abaykal20/TediGAN/base/attribute_classification/"
# image_folder = "/scratch/users/abaykal20/stylemc/fixed_attr/"
# image_folder = "/scratch/users/abaykal20/StyleCLIP/fixed_attr/"
# image_folder = "/scratch/users/abaykal20/StyleCLIP/global_directions/attr/"
experiment = "lighterer_new2/"

inference_images_path = "/scratch/users/abaykal20/LACE/FFHQ/prepare_models_data/label_indexes"
images_path = "/datasets/CelebA/Img/img_align_celeba/"

# all_files = []
# for (dirpath, dirnames, filenames) in os.walk(image_folder + experiment):
#     all_files += [os.path.join(dirpath, file) for file in filenames]
    
# print(len(all_files))
# print(all_files[0])
# print(all_files[51])
clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')

img_transforms = transforms.Compose([
            transforms.CenterCrop((178,178)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img_transforms2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

mps = []

for key, value in tqdm(captions_dict.items()):
    inference_path = os.path.join(inference_images_path, key + ".txt")
    f2 = open(inference_path, "r")
    caption = value
    for i in range(50):
        complete_image_path = os.path.join(images_path, f2.readline().rstrip())
        original_image = Image.open(complete_image_path).convert("RGB")
        input_image = img_transforms(original_image)
        
        manipulated_image_path = image_folder + experiment + key + "/" + f"{i}.jpg"
        manipulated_image = Image.open(manipulated_image_path).convert("RGB")
        manipulated_image_tensor = img_transforms2(manipulated_image)
        l1_diff = (torch.sum(torch.abs(input_image - manipulated_image_tensor)) / (256*256*3)).item()
        
        manipulated_image = clip_preprocess(manipulated_image).unsqueeze(0).cuda()
        text = clip.tokenize(caption).cuda()
        
        with torch.no_grad():
            logit, _ = clip_model(manipulated_image, text)
            clip_sim = logit[0,0].item() / clip_model.logit_scale.exp().item()
        mp = clip_sim * (1.0 - l1_diff)
        mps.append(mp)
print("Mean MP:", np.mean(mps))
        
    


