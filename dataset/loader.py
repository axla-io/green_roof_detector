import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from dataclasses import dataclass
@dataclass
class LoaderOptions:
    """Class for storing data loader options. 
    Currently we enforce all images to be monochrome and normalized and square
    This class have the keyword arguments:
    - `augumentation`, which determines whether the data is augmented
    - `randomize_augmentation`, toggles whether augumentation is done randomly
    - `p_thresh`, probability threshold to perform augmentation
    - `fraction`, fraction of original dataset to use for testing
    - `w`, width/height of output image
    """
    augmentation: bool = False #TODO: Implement functionality
    randomize_augmentation: bool = False #TODO: Implement functionality
    p_thresh: float = 0.5 #TODO: Implement functionality
    fraction: float = 0.25 
    w: int = 256
class Loader:
    def __init__(self, options) -> None:
        self.options = options
    
    def _get_split_prefix(self, index, test_length):
        return "test" if index < test_length else "train"
    
    def clean_dir(self, dir):
        for dirpath, dirnames, filenames in os.walk(dir):
            for file in filenames:
                os.remove(os.path.join(dirpath, file))
    
    def preprocess(self, in_dir, out_dir, file_names) -> None:
        # Shuffle the dataset and set limit for number of test images
        random.shuffle(file_names)
        data_length = len(file_names)
        test_length = int(round(self.options.fraction * data_length))

        for i,file_name in enumerate(file_names):
            print(f"Processing file: {i+1} of {data_length}")

            # Preprocess images
            img_path_in = os.path.join(in_dir, "images", file_name)
            img = self.preprocess_img(img_path_in)

            # Preprocess ground truth mask
            gt_path_in = os.path.join(in_dir, "masks", file_name)
            mask = self.preprocess_mask(gt_path_in)

            # Split dataset
            prefix = self._get_split_prefix(i, test_length)

            # Get output paths
            img_path_out = os.path.join(out_dir, prefix,"images", file_name)
            gt_path_out = os.path.join(out_dir, prefix,"masks", file_name)

            # Output processed files
            cv2.imwrite(img_path_out, img)
            cv2.imwrite(gt_path_out, mask)
    
    def preprocess_img(self, file_path) -> None:
        # Preprocess data
        train_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        small_img = cv2.resize(train_img, (self.options.w,self.options.w)) # Resize first to make subsequent operations faster
        normalized_gray_img = cv2.normalize(small_img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return normalized_gray_img
    
    def preprocess_mask(self, file_path) -> None:
        gt_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        small_img = cv2.resize(gt_img, (self.options.w,self.options.w))
        binary_mask = (small_img > 127).astype(np.uint8)
        return binary_mask

if __name__ == "__main__":
    options = LoaderOptions(w = 256)
    loader = Loader(options)
    in_dir = "data/original"
    out_dir = "data/processed"
    loader.clean_dir(out_dir)
    file_names = os.listdir(os.path.join(in_dir,"images"))
    loader.preprocess(in_dir, out_dir, file_names)
