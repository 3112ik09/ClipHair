import os
import numpy as np
from PIL import Image

inference_images_path = "/scratch/users/abaykal20/LACE/FFHQ/prepare_models_data/label_indexes"
images_path = "/datasets/CelebA/Img/img_align_celeba/"
captions_dict = {"Blond_Hair" : "This person has blond hair.",
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

images_path1 = "attribute_classification/hairclip_modified/"
images_path2 = "attribute_classification/cycle/"
images_path3 = "attribute_classification/perceptual/"

width, height = 178,218  # Get dimensions

left = (width - 178)/2
top = (height - 178)/2
right = (width + 178)/2
bottom = (height + 178)/2

for key,value in captions_dict.items():
    inference_path = os.path.join(inference_images_path, key + ".txt")
    f2 = open(inference_path, "r")
    images_path_dir1 = images_path1 + key
    images_path_dir2 = images_path2 + key
    images_path_dir3 = images_path3 + key
    if not os.path.isdir("attribute_merged/" + key):
        os.mkdir("attribute_merged/" + key)
    for i in range(10):
        complete_image_path = os.path.join(images_path, f2.readline().rstrip())
        original_image = Image.open(complete_image_path)
        complete_path1 = os.path.join(images_path_dir1, f'{i}.jpg')
        complete_path2 = os.path.join(images_path_dir2, f'{i}.jpg')
        complete_path3 = os.path.join(images_path_dir3, f'{i}.jpg')
        original_image = original_image.crop((left, top, right, bottom))
        original_image = original_image.resize((1024,1024))
        image1 = Image.open(complete_path1)
        image2 = Image.open(complete_path2)
        image3 = Image.open(complete_path3)
        results = np.concatenate([original_image, image1, image2, image3], axis=1)
        results = Image.fromarray(results)
        results.save(f"attribute_merged/{key}/{i}.jpg")
    
# images_path1 = "/scratch/users/abaykal20/sam/SAM/mmcelebhq/model_comparisons/"
# images_path2 = "inference_out_text/"
# images_path3 = "inference_out/"
# images_path4 = "inference_out_no_multiplier/"
# images_path5 = "inference_out_cycle/"

# print("Starting Merging")

# for i in range(50):
#     image1_path = images_path1 + f'{i}.png'
#     image2_path = images_path2 + f'{i}.png'
#     image3_path = images_path3 + f'{i}.png'
#     image4_path = images_path4 + f'{i}.png'
#     image5_path = images_path5 + f'{i}.png'
#     image1 = Image.open(image1_path)
#     image2 = Image.open(image2_path)
#     image3 = Image.open(image3_path)
#     image4 = Image.open(image4_path)
#     image5 = Image.open(image5_path)
#     results = np.concatenate([image1, image2, image3, image4, image5], axis=1)
#     results = Image.fromarray(results)
#     results.save("merged_all/{}.png".format(i))
    

print("Finished merging")