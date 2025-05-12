import os
import cv2
from jax import numpy as jnp
import numpy as np

def load_img(file_path):
    train_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    normalized_train_img = cv2.normalize(train_img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized_train_img

def load_imgs(file_paths):
    data = jnp.stack([load_img(path) for path in file_paths])
    data = jnp.expand_dims(data,3)
    return data

def get_filepaths(folder):
    return [os.path.join(folder, filename) for filename in os.listdir(folder)]

def load_data(main_dir):
    # Get folders
    x_train_dir = os.path.join(main_dir, "train/images")
    y_train_dir = os.path.join(main_dir, "train/masks")

    x_valid_dir = os.path.join(main_dir, "test/images")
    y_valid_dir = os.path.join(main_dir, "test/masks")
    # Import images
    x_train = load_imgs(get_filepaths(x_train_dir))
    y_train = load_imgs(get_filepaths(y_train_dir))

    x_valid = load_imgs(get_filepaths(x_valid_dir))
    y_valid = load_imgs(get_filepaths(y_valid_dir))

    return (x_train, y_train), (x_valid, y_valid)

def data_generator(images, labels, batch_size=128, is_valid=False, key=None):
    # 1. Calculate the total number of batches
    num_batches = int(np.ceil(len(images) / batch_size))

    # 2. Get the indices and shuffle them
    indices = np.arange(len(images))

    if not is_valid:
        if key is None:
             raise ValueError("A PRNG key is required if `aug` is set to True")
        else:
            np.random.shuffle(indices)

    for batch in range(num_batches):
        curr_idx = indices[batch * batch_size: (batch+1) * batch_size]
        batch_images = images[curr_idx]
        batch_labels = labels[curr_idx]
        yield batch_images, batch_labels

if __name__ == "__main__":
    main_dir = "data/processed"
    (x_train, y_train), (x_valid, y_valid) = load_data(main_dir)

    

