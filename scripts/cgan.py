import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, utils, metrics, optimizers

IMAGE_SIZE = 64
CHANNELS = 3
CLASSES = 2
BATCH_SIZE = 128
Z_DIM = 32
LEARNING_RATE = 0.00005
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
EPOCHS = 20
CRITIC_STEPS = 3
GP_WEIGHT = 10.0
LOAD_MODEL = False
LABEL = "Blond_Hair"

# 1.Prepare the data

attributes = pd.read_csv("../data/list_attr_celeba.csv")
print(attributes.columns)
attributes.head()

labels = attributes[LABEL].tolist()
int_labels = [x if x == 1 else 0 for x in labels]

train_data = utils.image_dataset_from_directory(
    "../data/img_align_celeba",
    labels=int_labels,
    color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)


def preprocessing(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img


train = train_data.map(lambda x, y:
                       (preprocessing(x), tf.one_hot(y, depth=CLASSES)))

# 2. Build the GAN

critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
label_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES))
x = layers.Concatenate(axis=-1)([critic_input, label_input])
x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(1, kernel_size=4, strides=1, padding='valid')(x)
critic_output = layers.Flatten()(x)

critic = models.Model([critic_input, label_input], critic_output)
critic.summary()

generator_input = layers.Input(shape=(Z_DIM, ))
label_input = layers.Input(shape=(CLASSES, ))
x = layers.Concatenate(axis=-1)([generator_input, label_input])
x = layers.Reshape((1, 1, Z_DIM + CLASSES))(x)
x = layers.Conv2DTranspose(128,
                           kernel_size=4,
                           strides=1,
                           padding='valid',
                           use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(128,
                           kernel_size=4,
                           strides=2,
                           padding='same',
                           use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(128,
                           kernel_size=4,
                           strides=2,
                           padding='same',
                           use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(64,
                           kernel_size=4,
                           strides=2,
                           padding='same',
                           use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(CHANNELS,
                                          kernel_size=4,
                                          strides=2,
                                          padding='same',
                                          activation='tanh')(x)
generator = models.Model([generator_input, label_input], generator_output)
generator.summary()


class ConditionalWGAN(models.Model):

    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(ConditionalWGAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(ConditionalWGAN, self).compile(run_eagerly=True)
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name='c_wass_loss')
        self.c_gp_metric = metrics.Mean(name='c_gp')
        self.c_loss_metric = metrics.Mean(name='c_loss')
        self.g_loss_metric = metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [
            self.c_loss_metric, self.c_wass_loss_metric, self.c_gp_metric,
            self.g_loss_metric
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images,
                         image_one_hot_labels):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images * alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels],
                               training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0)**2)
        return gp

    def train_step(self, data):
        real_images, one_hot_labels = data

        image_one_hot_labels = one_hot_labels[:, None, None, :]
        image_one_hot_labels = tf.repeat(image_one_hot_labels,
                                         repeats=IMAGE_SIZE,
                                         axis=1)
        image_one_hot_labels = tf.repeat(image_one_hot_labels,
                                         repeats=IMAGE_SIZE,
                                         axis=2)

        batch_size = tf.shape(real_images)[0]

        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                            self.latent_dim))

            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    [random_latent_vectors, one_hot_labels], training=True)
                fake_predictions = self.critic(
                    [fake_images, image_one_hot_labels], training=True)
                real_predictions = self.critic(
                    [real_images, image_one_hot_labels], training=True)
                c_wass_loss = tf.reduce_mean(
                    fake_predictions) - tf.reduce_mean(real_predictions)
                c_gp = self.gradient_penalty(batch_size, real_images,
                                             fake_images, image_one_hot_labels)
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                        self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images = self.generator(
                [random_latent_vectors, one_hot_labels], training=True)
            fake_predictions = self.critic([fake_images, image_one_hot_labels],
                                           training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss,
                                     self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables))
        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


cgan = ConditionalWGAN(critic=critic,
                       generator=generator,
                       latent_dim=Z_DIM,
                       critic_steps=CRITIC_STEPS,
                       gp_weight=GP_WEIGHT)

cgan.compile(c_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE,
                                         beta_1=ADAM_BETA_1,
                                         beta_2=ADAM_BETA_2),
             g_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE,
                                         beta_1=ADAM_BETA_1,
                                         beta_2=ADAM_BETA_2))

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.ckpt.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")


class ImageGenerator(callbacks.Callback):

    def __init__(self, num_img, latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img,
                                                        self.latent_dim))
        zero_label = tf.convert_to_tensor(
            np.repeat([[1, 0]], self.num_img, axis=0))
        generated_images = self.model.generator(
            [random_latent_vectors, zero_label])
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.numpy()

        one_label = tf.convert_to_tensor(
            np.repeat([[0, 1]], self.num_img, axis=0))
        generated_images = self.model.generator(
            [random_latent_vectors, one_label])
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.numpy()


history = cgan.fit(
    train,
    epochs=EPOCHS * 100,
    steps_per_epoch=1,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
        ImageGenerator(num_img=10, latent_dim=Z_DIM),
    ],
)

generator.save("./models/generator.keras")
critic.save('./models/critic.keras')

# Generate images

GROUP_SIZE = 10
OUTPUT_PATH = './output'


def normalize_img(images):
    assert imgs.shape == (GROUP_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    return [(i - i.min()) / (i.max() - i.min()) for i in images]


Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# 0 label
z_sample = np.random.normal(size=(GROUP_SIZE, Z_DIM))
class_label = np.repeat([[1, 0]], GROUP_SIZE, axis=0)
imgs = cgan.generator.predict([z_sample, class_label])
for idx, img in enumerate(normalize_img(imgs)):
    plt.imsave(f'output/cgan_generated_l0_{idx}.jpg', img)

# 1 label
z_sample = np.random.normal(size=(GROUP_SIZE, Z_DIM))
class_label = np.repeat([[0, 1]], GROUP_SIZE, axis=0)
imgs = cgan.generator.predict([z_sample, class_label])
for idx, img in enumerate(normalize_img(imgs)):
    plt.imsave(f'output/cgan_generated_l1_{idx}.jpg', img)
