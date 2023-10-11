import os
import numpy as np
from PIL import Image

images_path1 = "/scratch/users/abaykal20/sam/SAM/mmcelebhq/model_comparisons/"
images_path2 = "inference_out_text/"
images_path3 = "inference_out/"
images_path4 = "inference_out_no_multiplier/"
images_path5 = "inference_out_cycle/"

print("Starting Merging")

for i in range(50):
    image1_path = images_path1 + f'{i}.png'
    image2_path = images_path2 + f'{i}.png'
    image3_path = images_path3 + f'{i}.png'
    image4_path = images_path4 + f'{i}.png'
    image5_path = images_path5 + f'{i}.png'
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image3 = Image.open(image3_path)
    image4 = Image.open(image4_path)
    image5 = Image.open(image5_path)
    results = np.concatenate([image1, image2, image3, image4, image5], axis=1)
    results = Image.fromarray(results)
    results.save("merged_all/{}.png".format(i))
    

print("Finished merging")