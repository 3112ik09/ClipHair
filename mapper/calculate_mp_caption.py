import torch.nn as nn
from torchvision import transforms
import torch
import numpy as np
import os
import clip
from PIL import Image
from tqdm import tqdm

# captions_dict = {
#                  "Blond_Hair" : "This person has blond hair.",
#                  "Bushy_Eyebrows" : "This person has bushy eyebrows.",
#                  "Chubby" : "This person is chubby.",
#                  "Double_Chin" : "This person has double chin.",
#                  "Eyeglasses" : "This person has eyeglasses.",
#                  "Goatee" : "This person has a goatee.",
#                  "Gray_Hair" : "This person has gray hair.",
#                  "Heavy_Makeup" : "This person wears heavy makeup.",
#                  "Male" : "This is a male.",
#                  "Mouth_Slightly_Open" : "This person has mouth slightly open.",
#                  "Mustache" : "This person has a mustache.",
#                  "Rosy_Cheeks" : "This person has rosy cheeks.",
#                  "Smiling" : "This person is smiling.",
#                  "Wearing_Lipstick" : "This person is wearing lipstick.",
#                  "Wearing_Necktie" : "This person is wearing a necktie."}

captions_dict = {"Blond_Hair" : {"female" : "She has blond hair.", "male" : "He has blond hair."},
                 "Bushy_Eyebrows" : {"female" : "She has bushy eyebrows.", "male" : "He has bushy eyebrows."},
                 "Chubby" : {"female" : "She is chubby.", "male" : "He is chubby."},
                 "Double_Chin" : {"female" : "He has double chin.", "male" : "He has double chin."}, # changed
                 "Eyeglasses" : {"female" : "She has eyeglasses.", "male" : "He has eyeglasses."},
                 "Goatee" : {"female" : "He has goatee.", "male" : "He has goatee."},
                 "Gray_Hair" : {"female" : "She has gray hair.", "male" : "He has gray hair."},
                 "Heavy_Makeup" : {"female" : "She wears heavy makeup.", "male" : "She wears heavy makeup."},
                 "Male" : {"female" : "This is a male.", "male" : "This is a male."},
                 "Mouth_Slightly_Open" : {"female" : "She has mouth slightly open.", "male" : "He has mouth slightly open."},
                 "Mustache" : {"female" : "He has mustache.", "male" : "He has mustache."},
                 "Rosy_Cheeks" : {"female" : "She has rosy cheeks.", "male" : "She has rosy cheeks."}, # changed
                 "Smiling" : {"female" : "She is smiling.", "male" : "He is smiling."},
                 "Wearing_Lipstick" : {"female" : "She is wearing lipstick.", "male" : "She is wearing lipstick."},
                 "Wearing_Necktie" : {"female" : "He is wearing necktie.", "male" : "He is wearing necktie."}} # changed

label_path = "/datasets/CelebA/Anno/list_attr_celeba.txt"
label_list = open(label_path).readlines()[2:]
data_label = []
for i in range(len(label_list)):
    data_label.append(label_list[i].split())

# transform label into 0 and 1
for m in range(len(data_label)):
    data_label[m] = [n.replace('-1', '0') for n in data_label[m][1:]]
    data_label[m] = [int(p) for p in data_label[m]]
    
gender_index = 20

# image_folder = "/scratch/users/abaykal20/sam/SAM/attribute_classification/"
image_folder = "attribute_classification/"
# image_folder = "/scratch/users/abaykal20/TediGAN/base/attribute_classification/"
# image_folder = "/scratch/users/abaykal20/stylemc/fixed_attr/"
# image_folder = "/scratch/users/abaykal20/StyleCLIP/fixed_attr/"
# image_folder = "/scratch/users/abaykal20/StyleCLIP/global_directions/attr/"
experiment = "hairclip_modified_new2/"

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
    # caption = value
    for i in range(50):
        img_idx = f2.readline().rstrip()
        img_idx_int = int(img_idx.lstrip("0").rstrip(".jpg")) - 1
        if data_label[img_idx_int][gender_index] == 0:
            gender = "female"
        else:
            gender = "male"
        caption = value[gender]
        
        complete_image_path = os.path.join(images_path, img_idx)
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
        
    


