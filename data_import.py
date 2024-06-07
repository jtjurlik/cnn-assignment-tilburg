# The majority of this code was prepared and made available by the teacher of 
# the Deep Learning course (880008-M-6) in the Spring semester 2024, dr. Görkem 
# Saygili PhD.

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ipython_config

# Replace these paths with the actual paths to your dataset folders
data_dir = ipython_config.DATA_PATH
filepaths = []
image_data = []
labels = []

folds = os.listdir(data_dir)

for fold in folds:
    foldpath = os.path.join(data_dir, fold)

    if os.path.isdir(foldpath):
        flist = os.listdir(foldpath)

        for f in flist:
            f_path = os.path.join(foldpath , f)

            # Check if f_path is a directory
            if os.path.isdir(f_path):
                filelist = os.listdir(f_path)

                for file in filelist:
                    fpath = os.path.join(f_path , file)
                    try:
                    # Open the image using PIL (or you can use OpenCV) within a 'with' statement
                        with Image.open(fpath) as image:
                            if image is not None:
                                # Resize images
                                im = image.resize((120,120), Image.LANCZOS)
                                # Append image and label to respective lists
                                image_data.append(np.array(im))
                            else:
                                print(f"Error opening image '{fpath}': NoneType object returned")
                    except Exception as e:
                        print(f"Error opening image '{fpath}': {e}")
                    # Assign the label of the images according to the folder they belongs to.
                    if f == 'colon_aca':
                        labels.append('Colon adenocarcinoma')

                    elif f == 'colon_n':
                        labels.append('Colon Benign Tissue')

                    elif f == 'lung_aca':
                        labels.append('Lung adenocarcinoma')

                    elif f == 'lung_n':
                        labels.append('Lung Benign Tissue')

                    elif f == 'lung_scc':
                        labels.append('Lung Squamous Cell Carcinoma')
            else:
                print(f"Skipping {f_path} as it is not a directory.")
    else:
        print(f"Skipping {foldpath} as it is not a directory.")

tr_labels = np.array(labels)
image_matrix = np.array([np.array(img) for img in image_data])

np.save("data120.npy",image_matrix)
np.save("labels120.npy",tr_labels)
