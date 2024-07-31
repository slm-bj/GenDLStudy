import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils, metrics, optimizers

IMAGE_SIZE = 64
CHANNELS = 3
CLASSES = 2
MATCH_SIZE = 128
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
