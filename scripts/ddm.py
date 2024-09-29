import math
import tensorflow as tf
from tensorflow.keras import utils
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

# # The Reverse Diffusion Process

