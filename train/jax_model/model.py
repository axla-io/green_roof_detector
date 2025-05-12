import jax
import jax.numpy as jnp
from jax import lax
import optax
from flax import linen as nn
from flax.training import train_state
from jax_model.custom_layer import *

class ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, strides=None, padding = 0):
        x = nn.Conv(self.features, (3, 3), strides=strides , padding=padding)(x)
        x = nn.relu(x)
        return x

class UNetWithAttention(nn.Module):
    base_features: int = 64

    @nn.compact
    def __call__(self, x):
        # Encoder
        x1 = ConvBlock(self.base_features)(x, strides = 2, padding = 2)
        x1_con = nn.max_pool(x1, (2, 2))
        x2 = ConvBlock(self.base_features * 2)(x1_con, strides = 2, padding = 2)
        x2_con = nn.max_pool(x2, (2, 2))

        # Bottleneck with self-attention only
        x3 = ConvBlock(self.base_features * 4)(x2_con, strides = 2, padding = 1)
        x3 = SelfAttentionBlock(num_heads=4, qkv_features=self.base_features * 4)(x3)

        # Decoder
        x = nn.ConvTranspose(self.base_features * 2, (2, 2), strides=(2, 2))(x3)
        x = jnp.concatenate([x, x2_con], axis=-1)
        x = ConvBlock(self.base_features * 2)(x,padding = "SAME")

        x = nn.ConvTranspose(self.base_features, (2, 2), strides=(2, 2))(x)
        x = jnp.concatenate([x, x1_con], axis=-1)
        x = ConvBlock(self.base_features)(x,padding = "SAME")

        # Output layer
        x = nn.ConvTranspose(self.base_features, (2, 2), strides=(2, 2))(x)
        x = nn.Conv(1, (1, 1))(x)
        return nn.sigmoid(x)  # for binary segmentation

# B. Loss function we want to use for the optimization
def calculate_loss(params, inputs, labels):
    """Cross-Entropy loss function.

    Args:
        params: The parameters of the model at the current step
        inputs: Batch of images
        labels: Batch of corresponding labels
    Returns:
        loss: Mean loss value for the current batch
        logits: Output of the last layer of the classifier
    """
    logits = UNetWithAttention().apply({'params': params}, inputs)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))
    return loss, logits

# C. Evaluation metric
def calculate_accuracy(logits, labels):
    return  1.0 - jnp.mean(abs(logits - labels))

# D. Train step. We will jit transform it to compile it. We will get a
# good speedup on the subseuqent runs
@jax.jit
def train_step_cpu(state, batch_data):
    # 1. Get the images and the labels
    inputs, labels = batch_data

    # 2. Calculate the loss and get the gradients
    (loss, logits), grads = jax.value_and_grad(calculate_loss, has_aux=True)(state.params, inputs, labels)

    # 3. Calculate the accuracy for the cuurent batch
    accuracy = calculate_accuracy(logits, labels)

    # 4. Update the state (parameters values) of the model
    state = state.apply_gradients(grads=grads)

    # 5. Return loss, accuracy and the updated state
    return loss, accuracy, state

# E. Test/Evaluation step. We will jit transform it to compile it as well.
@jax.jit
def test_step_cpu(state, batch_data):
    # 1. Get the images and the labels
    inputs, labels = batch_data

    # 2. Calculate the loss
    loss, logits = calculate_loss(state.params, inputs, labels)

    # 3. Calculate the accuracy
    accuracy = calculate_accuracy(logits, labels)

    # 4. Return loss and accuracy values
    return loss, accuracy

@jax.pmap(axis_name="batch")
def train_step(state, batch_data):
    inputs, labels = batch_data

    def loss_fn(params):
        (loss, logits) = calculate_loss(params, inputs, labels)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Aggregate gradients across devices
    grads = lax.pmean(grads, axis_name="batch")

    # Aggregate loss and accuracy
    loss = lax.pmean(loss, axis_name="batch")
    accuracy = lax.pmean(calculate_accuracy(logits, labels), axis_name="batch")

    # Apply gradients
    state = state.apply_gradients(grads=grads)

    return loss, accuracy, state

@jax.pmap(axis_name="batch")
def test_step(state, batch_data):
    inputs, labels = batch_data

    loss, logits = calculate_loss(state.params, inputs, labels)

    # Aggregate across devices
    loss = lax.pmean(loss, axis_name="batch")
    accuracy = lax.pmean(calculate_accuracy(logits, labels), axis_name="batch")

    return loss, accuracy

def shard(batch, num_devices):
    # Assumes batch is a tuple of (x, y)
    x, y = batch
    x = x.reshape((num_devices, -1) + x.shape[1:])  # [num_devices, local_batch, ...]
    y = y.reshape((num_devices, -1) + y.shape[1:])
    return x, y

# F. Initial train state including parameters initialization
def create_train_state(key, lr=1e-4):
    """Creates initial `TrainState for our classifier.

    Args:
        key: PRNG key to initialize the model parameters
        lr: Learning rate for the optimizer

    """
    # 1. Model instance
    model = UNetWithAttention()

    # 2. Initialize the parameters of the model
    params = model.init(key, jnp.ones([1, 256, 256, 1]))['params']

    # 3. Define the optimizer with the desired learning rate
    optimizer = optax.adam(learning_rate=lr)

    # 4. Create and return initial state from the above information. The `Module.apply` applies a
    # module method to variables and returns output and modified variables.
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
