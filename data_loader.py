# data_loader.py
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def load_data_from_directory(base_path, categories, image_size=(150, 150)):
    data, labels = [], []
    for label_index, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        for img_file in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_file)
                img = imread(img_path)
                img_resized = resize(img, image_size)
                data.append(img_resized.flatten())
                labels.append(label_index)
            except:
                print(f"Failed to load: {img_path}")
    return np.array(data), np.array(labels)
