import math
import keras
import tensorflow as tf
from tensorflow.keras import activations, callbacks, layers, losses, models, metrics, optimizers, utils
from utils import display, sample_batch

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
LOAD_MODEL = False

NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 20

EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50

# # The Flowers Dataset

train_data = utils.image_dataset_from_directory(
        "../data/pytorch-challange-flower-dataset/dataset",
        labels=None,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=None,
        shuffle=True,
        seed=42,
        interpolation="bilinear")

def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img

train = train_data.map(lambda x: preprocess(x))
train = train.repeat(DATASET_REPETITIONS)
train = train.batch(BATCH_SIZE, drop_remainder=True)

train_sample = sample_batch(train)
display(train_sample)

# # The Forward Diffusion Process

# > This is exactly what we need, as we want to be able to easily sample $x_T$ and
# > then apply a **reverse diffusion process** through our trained neural network model!

# > We can define a function q that adds a small amount of Gaussian noise with
# variance $\beta_t$ to an image $x_{t − 1}$ to generate a new image $x_t$ ...

# But in the following equation, $x_t$ is written as a independent variable rather
# than a dependent variable:

# $$
# q(x_t \vert x_{t-1}) = \mathscr{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbb{I})
# $$

# According to [this answer](https://math.stackexchange.com/questions/3421665/do-the-vertical-bar-and-semicolon-denote-the-same-thing-in-the-context-of-condit),
# $f(x, y, z)$ means a function $f$ with arguments $x$, $y$ and $z$.
# While $f(x; y, z)$ means function $f$ with **fixed** values of $y$ and $z$,
# while we study only the relationship between the function value and $x$.
# Here $y$ and $z$ are called **parameter** rather than **argument**.

# # The Reparameterization Trick

# Prove of reparameterization equation at the top of page 211:

# $$
# \begin{equation}
# \begin{split}
# x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
# &= \sqrt{\alpha_t} \left(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1- \alpha_{t-1}} \epsilon_{t-2} \right) +  \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
# &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t - \alpha_t \alpha_{t-1}} \epsilon_{t-2} +  \sqrt{1 - \alpha_t} \epsilon_{t-1} 
# \end{split}
# \end{equation}
# $$
# $$
# \because Var(\sqrt{\alpha_t - \alpha_t \alpha_{t-1}} \epsilon_{t-2}) = \alpha_t - \alpha_t \alpha_{t-1} \\
# Var(\sqrt{1 - \alpha_t} \epsilon_{t-1}) = 1 - \alpha_t \\
# \therefore Var(\sqrt{\alpha_t - \alpha_t \alpha_{t-1}} \epsilon_{t-2} +  \sqrt{1 - \alpha_t} \epsilon_{t-1} ) = 1 - \alpha_t \alpha_{t-1} \\
# \therefore \sqrt{\alpha_t - \alpha_t \alpha_{t-1}} \epsilon_{t-2} +  \sqrt{1 - \alpha_t} \epsilon_{t-1}  = \sqrt{1 - \alpha_t \alpha_{t-1}} \epsilon \\
# \begin{equation}
# \begin{split}
# \therefore x_t &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \epsilon \\
# &= \cdots \\
# &= \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
# \end{split}
# \end{equation}
# $$

# Notes: distinguish *sum of normally distributed random variables* and 
# *sum of normal distributions*:
#
# * https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
# * https://en.wikipedia.org/wiki/Mixture_distribution
# * https://stats.stackexchange.com/questions/125808/sum-of-gaussian-is-gaussian

# # Diffusion Schedules

# > How the $\beta_t$ (or $\alpha_t$) values change with $t$ is called
# > the *diffusion schedule*.

def linear_diffusion_schedule(diffusion_times):
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = tf.math.cumprod(alphas)
    signal_rates = tf.sqrt(alpha_bars)
    noise_rates = tf.sqrt(1 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    signal_rates = tf.cos(diffusion_times * math.pi / 2)
    noise_rates = tf.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates

# # The U-Net Denoising Model

# ## Sinusoidal Embedding

def sinusoidal_embedding(x):
    frequencies = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), 16,))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings

# ## Residual Block

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding='same', activation=activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
        x = layers.Add()([x, residual])
        return x
    return apply

# ## DownBlocks and UpBlocks

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply


# The U-Net denoising model:

noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = layers.Conv2D(32, kernel_size=1)(noisy_images)

noise_variances = layers.Input(shape=(1, 1, 1))
noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
noise_embedding = layers.UpSampling2D(size=64, interpolation="nearest")(noise_embedding)

x = layers.Concatenate()([x, noise_embedding])

skips = []

x = DownBlock(32, block_depth=2)([x, skips])
x = DownBlock(64, block_depth=2)([x, skips])
x = DownBlock(96, block_depth=2)([x, skips])

x = ResidualBlock(128)(x)
x = ResidualBlock(128)(x)

x = UpBlock(96, block_depth=2)([x, skips])
x = UpBlock(64, block_depth=2)([x, skips])
x = UpBlock(32, block_depth=2)([x, skips])

x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)
unet = models.Model([noisy_images, noise_variances], x, name="unet")

# # The Reverse Diffusion Process

@keras.saving.register_keras_serializable()
class DiffusionModel(models.Model):
    def __init__(self, nw):
        super().__init__()
        self.normalizer = layers.Normalization()
        self.network = nw  # described in section "The U-Net Denoising Model"
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = cosine_diffusion_schedule

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noise = network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noise) / signal_rates
        return pred_noise, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)  # $x_0$ (line no.2) in figure 8-7
        noises = tf.random.normal(shape=tf.shape(images))  # $\epsilon$ (line no.4) in figure 8-7
        batch_size = tf.shape(images)[0]
        diffusion_times = tf.random.uniform(
                shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)  # $t$ (line no.3) in figure 8-7
        noise_rates, signal_rates = self.cosine_diffusion_schedule(diffusion_times)  # $\alpha_t$ in figure 8-7
        noisy_images = signal_rates * images + noise_rates * noises  # $x_t$ define at the top of page 211

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(  # pred_noises is $\epsilon_\theta$ in figure 8-7
                    noisy_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)  # line no.5 in figure 8-7
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(0.999 * ema_weight + (1 - 0.999) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.random.uniform(shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)
        return {m.name: m.result() for m in self.metrics}



# # Training the Diffusion Model

ddm = DiffusionModel(unet)
ddm.normalizer.adapt(train)

if LOAD_MODEL:
    ddm.built = True
    ddm.load_weights("./checkpoint/ddm-checkpoints.ckpt")

ddm.compile(optimizer=optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=losses.mean_absolute_error)
model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="./checkpoint/ddm-checkpoints.ckpt",
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generate(
            num_images=self.num_img,
            diffusion_steps=PLOT_DIFFUSION_STEPS,
        ).numpy()
        display(
            generated_images,
            save_to="./output/ddm-generated_img_%03d.png" % (epoch),
        )

image_generator_callback = ImageGenerator(num_img=10)

ddm.fit(train,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            image_generator_callback,
        ],)
