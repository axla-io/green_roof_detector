import os
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
    
    def preprocess(self, in_dir, out_dir, file_names) -> None:
        file_name = file_names[0]
        # Get indices to use in testing dataset
        data_length = len(file_names)
        test_length = int(round(self.options.fraction * data_length))
        train_length = data_length - test_length
        
        # Preprocess images
        train_path = os.path.join(in_dir, "images", file_name)
        self.preprocess_img(train_path)

        # Preprocess ground truth
        gt_path = os.path.join(in_dir, "masks", file_name)
        self.preprocess_mask(gt_path)


        # Output image to 
       
    
    def preprocess_img(self, file_path) -> None:
        # Preprocess data
        train_img = cv2.imread(file_path)
        small_img = cv2.resize(train_img, (self.options.w,self.options.w)) # Resize first to make subsequent operations faster
        gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        normalized_gray_img = cv2.normalize(gray_img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        print(normalized_gray_img.shape)

    
    def preprocess_mask(self, file_path) -> None:
        gt_img = cv2.imread(file_path)
        small_img = cv2.resize(gt_img, (self.options.w,self.options.w))
        binary_mask = (small_img > 127).astype(np.uint8)
        print(binary_mask.shape)

if __name__ == "__main__":
    options = LoaderOptions(w = 128)
    loader = Loader(options)
    in_dir = "data/original"
    out_dir = "data"
    file_names = os.listdir(os.path.join(in_dir,"images"))
    loader.preprocess(in_dir, out_dir, file_names)
