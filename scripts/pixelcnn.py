import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.python.ops.custom_gradient import generate_name
from tensorflow.python.ops.gen_math_ops import Imag

# %% [markdown]
# # 0. Parameters

IMAGE_SIZE = 16
PIXEL_LEVELS = 4
N_FILTERS = 128
RESIDUAL_BLOCKS = 5
BATCH_SIZE = 128
EPOCHS = 150

# %% [markdown]
# # 1. Prepare the Data

# %%
(x_train, _), (_, _) = datasets.fashion_mnist.load_data()


def preprocess(imgs_int):
    imgs_int = np.expand_dims(imgs_int, -1)
    imgs_int = tf.image.resize(imgs_int, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    imgs_int = (imgs_int / (256 / PIXEL_LEVELS)).astype(int)
    imgs = imgs_int.astype("float32")
    imgs = imgs / PIXEL_LEVELS
    return imgs, imgs_int


input_data, output_data = preprocess(x_train)

input_data.shape  # (60000, 16, 16, 1)
output_data.shape  # (60000, 16, 16, 1)

# %% [markdown]
# # 2. Build the PixelCNN


# %%
class MaskedConv2D(layers.Layer):

    def __init__(self, mask_type, **kwargs):
        super(MaskedConv2D, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        # kernel_shape = self.conv.kernel.get_shape()
        kernel_shape = self.conv.kernel.shape
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[:kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == 'B':
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

    def get_config(self):
        return super().get_config()


# %%
class ResidualBlock(layers.Layer):

    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters=filters // 2,
                                   kernel_size=1,
                                   activation="relu")
        self.pixel_conv = MaskedConv2D(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = layers.Conv2D(filters=filters,
                                   kernel_size=1,
                                   activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return layers.add([inputs, x])

    def get_config(self):
        return super().get_config()


# %%
inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
x = MaskedConv2D(
    mask_type="A",
    filters=N_FILTERS,
    kernel_size=7,
    activation="relu",
    padding="same",
)(inputs)

for _ in range(RESIDUAL_BLOCKS):
    x = ResidualBlock(filters=N_FILTERS)(x)

for _ in range(2):
    x = MaskedConv2D(
        mask_type="B",
        filters=N_FILTERS,
        kernel_size=1,
        strides=1,
        activation='relu',
        padding='valid',
    )(x)

out = layers.Conv2D(
    filters=PIXEL_LEVELS,
    kernel_size=1,
    strides=1,
    activation='softmax',
    padding='valid',
)(x)

pixel_cnn = models.Model(inputs, out)
pixel_cnn.summary()

# %% [markdown]
# # 3. Train the PixelCNN

# %%
adam = optimizers.Adam(learning_rate=0.0005)
pixel_cnn.compile(optimizer=adam, loss="sparse_categorical_crossentropy")

# %%
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")


class ImageGenerator(callbacks.Callback):

    def __init__(self, num_img):
        self.num_lmg = num_img

    def sample_from(self, probs, temperature):
        pbs = probs**(1 / temperature)
        pbs = pbs / np.sum(pbs)
        return np.random.choice(len(pbs), p=pbs)

    def generate(self, temperature):
        generated_images = np.zeros(shape=(self.num_lmg, ) +
                                    (pixel_cnn.input_shape)[1:])
        batch, rows, cols, channels = generated_images.shape
        for row in range(rows):
            for col in range(cols):
                for channel in range(channels):
                    probs = self.model.predict(generated_images,
                                               verbose=0)[:, row, col, :]
                    generated_images[:, row, col, channel] = [
                        self.sample_from(x, temperature) for x in probs
                    ]
                    generated_images[:, row, col, channel] /= PIXEL_LEVELS
        return generated_images

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate(temperature=1.0)
        plt.imsave(f'./output/generated_img_{epoch}.png', generated_images)


img_generated_callback = ImageGenerator(num_img=10)

pixel_cnn.fit(
    input_data,
    output_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, img_generated_callback],
)

generated_images = img_generated_callback.generate(temperature=1.0)
