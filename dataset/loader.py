import os
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
@dataclass
class LoaderOptions:
    """Class for storing data loader options. 
    Currently we enforce all images to be monochrome and normalized and square
    This class have the keyword arguments:
    - `augumentation`, which determines whether the data is augmented
    - `randomize_augmentation`, toggles whether augumentation is done randomly
    - `p_thresh`, probability threshold to perform augmentation
    - `w`, width/height of output image
    """
    augmentation: bool
    randomize_augmentation: bool
    p_thresh: float
    w: int
class Loader:
    def __init__(self, options) -> None:
        self.options = options
    
    def preprocess(self, in_dir, out_dir, file_names) -> None:
        # Get indices to use in testing dataset
        data_length = len(file_names)
        test_length = int(round(self.options.fraction * data_length))
        train_length = data_length - test_length
        
        # Preprocess images
        train_path = os.path.join(train_dir, file_name)

        # Preprocess ground truth
        gt_path = os.path.join(gt_dir, file_name)

        # Output image to 
       
    
    def preprocess(self, train_dir, gt_dir, file_name) -> None:
        # Preprocess data
        train_path = os.path.join(train_dir, file_name)
        train_img = cv2.imread(train_path)
        cv2.resize(train_img, (self.options.w,self.options.w)) # Resize first to make subsequent operations faster
        gray_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        cv2.normalize(gray_img)
        print(gray_img)

        # Preprocess ground truth
        gt_path = os.path.join(gt_dir, file_name)
        gt_img = cv2.imread(gt_path)
        binary_mask = (gt_img > 127).astype(np.uint8)
        print(gt_img)
    
    def preprocess(self, train_dir, gt_dir, file_name) -> None:
        # Preprocess data
        train_path = os.path.join(train_dir, file_name)
        train_img = cv2.imread(train_path)
        cv2.resize(train_img, (self.options.w,self.options.w)) # Resize first to make subsequent operations faster
        gray_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        cv2.normalize(gray_img)
        print(gray_img)

        # Preprocess ground truth
        gt_path = os.path.join(gt_dir, file_name)
        gt_img = cv2.imread(gt_path)
        binary_mask = (gt_img > 127).astype(np.uint8)
        print(gt_img)

if __name__ == "__main__":
    options = LoaderOptions(w = 128)
    loader = Loader(options)
    in_dir = "data/original"
    out_dir = "data"
    file_names = os.listdir(os.path.join(in_dir,"images"))
    file_name = file_names[0]
    loader.preprocess(in_dir, out_dir, file_name)
