import os
import pandas as pd
import numpy as np

import torch

from PIL import Image
from torch.utils.data import Dataset

class PascalVOCLabelLoader:
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', cat_name])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        return df

    
class PascalVOCDataset(Dataset):
#     def __init__(self, img_root, ins_label_pairs , crop_size, transform=None):
    def __init__(self, img_root, classes, pvc, dataset_type, transform=None):
        
        """
        
        img_root: contains the path to the image root folder
        ins_label_pairs: instance label pair that contains a list of all the image path names and their respective labels
        crop_size: contains desired crop dimensions
        transform: contains the transformation procedures to be applied. defaulted to be None
        
        """
        
        self.img_root = img_root
        self.transform = transform
        self.classes = classes
        self.pvc = pvc
        self.dataset_type = dataset_type
        self.ins_label_pairs = self.instance_label_prep(self.classes, self.pvc, self.dataset_type)
  
    def __len__(self):
        return len(self.ins_label_pairs)
    
    def image_load(self, image_path):
        # Open image and load
        img = (Image.open(image_path))
        img.load()
        
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
            
        return Image.fromarray(img)
        
    def __getitem__(self, index):
        # Path to the image
        image_path = self.img_root + self.ins_label_pairs[index][0] + '.jpg'
        
        # Open the image
        image = self.image_load(image_path)
        label = torch.from_numpy((self.ins_label_pairs[index][1]).astype(float))
        
        if self.transform is not None:
            image = self.transform(image)
        
        return [image, label, self.ins_label_pairs[index][0]]
    
    def instance_label_prep(self, classes, pvc, dataset_type):
        
        """

        classes: a list containing the classes used
        pvc: pascalVOC object
        dataset_type: train, trainval or val

        """

        # Get a dataframe from within the pascalVOC dataset. It will be in a one hot encoding fashion
        final_df = None

        # Loop through each different class to get each image's classes
        for index, x in enumerate(classes):
            cat_name = x # category name

            df = pvc.imgs_from_category_as_list(cat_name, dataset_type)
            df[x] = df[x].replace(-1, 0)

            # For the first category, we use its dataframe as the base
            if index == 0:
                final_df = df
            # And merge with the rest of the following categories
            else:        
                final_df = final_df.merge(df, on='filename', how='inner')

        # Here we get each image name and their respective labels (one hot encoding format) and store in a list
        df_np = final_df.to_numpy()

        ins_labels = []
        for x in df_np:
            ins_labels.append([x[0], x[1:]])
        print

        return ins_labels


class ImageOnlyDataset(Dataset):
    def __init__(self, img_root, img_instances, transform=None):
        
        """
        img_root: contains the path to the image root folder
        ins_label_pairs: instance label pair that contains a list of all the image path names and their respective labels
        crop_size: contains desired crop dimensions
        transform: contains the transformation procedures to be applied. defaulted to be None
        
        """
        self.img_root = img_root
        self.img_instances = img_instances
        self.transform = transform
        
    def __len__(self):
        return len(self.img_instances)
    
    def image_load(self, image_path):
        # Open image and load
        img = (Image.open(image_path))
        img.load()
        
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
            
        return Image.fromarray(img)

    def __getitem__(self, index):
        # Path to the image
        image_path = self.img_root + self.img_instances[index] + '.jpg'
        
        # Open the image
        image = self.image_load(image_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return [image]