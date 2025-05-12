from jax_model.custom_layer import *
from jax_model.model import *
from jax import random
from load import load_data, data_generator
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 100
BATCH_SIZE = 10

# Initialize model
key = random.PRNGKey(0)
key, init_key = random.split(key)
state = create_train_state(init_key, lr=3*1e-5)

# Load training and testing set
main_dir = "data/processed"
(x_train, y_train), (x_valid, y_valid) = load_data(main_dir)

# Lists to record loss and accuracy for each epoch
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Training
for i in range(EPOCHS):
    num_train_batches = len(x_train) // BATCH_SIZE
    num_valid_batches = len(x_valid) // BATCH_SIZE

    # Lists to store loss and accuracy for each batch
    train_batch_loss, train_batch_acc = [], []
    valid_batch_loss, valid_batch_acc = [], []

    # Key to be passed to the data generator for augmenting
    # training dataset
    key, subkey = random.split(key)

    # Initialize data generators
    train_data_gen = data_generator(x_train,
                                y_train,
                                batch_size=BATCH_SIZE,
                                is_valid=False,
                                key=key
                               )

    valid_data_gen = data_generator(x_valid,
                               y_valid,
                               batch_size=BATCH_SIZE,
                               is_valid=False,
                               key=key
                               )

    print(f"Epoch: {i+1:<3}", end=" ")

    # Training
    for step in range(num_train_batches):
        batch_data = next(train_data_gen)
        loss_value, acc, state = train_step(state, batch_data)
        train_batch_loss.append(loss_value)
        train_batch_acc.append(acc)

    # Evaluation on validation data
    for step in range(num_valid_batches):
        batch_data = next(valid_data_gen)
        loss_value, acc = test_step(state, batch_data)
        valid_batch_loss.append(loss_value)
        valid_batch_acc.append(acc)

    # Loss for the current epoch
    epoch_train_loss = np.mean(train_batch_loss)
    epoch_valid_loss = np.mean(valid_batch_loss)

    # Accuracy for the current epoch
    epoch_train_acc = np.mean(train_batch_acc)
    epoch_valid_acc = np.mean(valid_batch_acc)

    training_loss.append(epoch_train_loss)
    training_accuracy.append(epoch_train_acc)
    validation_loss.append(epoch_valid_loss)
    validation_accuracy.append(epoch_valid_acc)

    print(f"loss: {epoch_train_loss:.3f}   acc: {epoch_train_acc:.3f}  valid_loss: {epoch_valid_loss:.3f}  valid_acc: {epoch_valid_acc:.3f}")
