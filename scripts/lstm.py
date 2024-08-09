# %%
from enum import UNIQUE
import json
import re
import string
from threading import activeCount
import tensorflow as tf
import numpy as np
from tensorflow.keras import callbacks, layers, losses, models

VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100  # how long is the vector in embedding space for each token
SEQ_SIZE = 200  # number of tokens (words) in a sequence
UNIT_SIZE = 128  # number of units in a LSTM cell

# %% [markdown]
# # The Recipe Dataset


# %%
def pad_punctuation(s):
    '''Add space before and after punctuations

    This action make punctuation a standalone word.
    But for some "informal" words the reulst of this translation is not godd enough.
    For example: the word "1/2" are transformed to three words: 1 / 2.
    '''
    s = re.sub(f'([{string.punctuation}])', r' \1 ', s)
    s = re.sub(' +', ' ', s)
    return s


# %% Import data, convert to tf.data.Dataset
with open('../data/full_format_recipes.json') as jd:
    recipe_data = json.load(jd)

filtered_data = [
    f"Recipe for {x['title']} | {' '.join(x['directions'])}"
    for x in recipe_data if 'title' in x and x['title'] is not None
    and 'directions' in x and x['directions'] is not None
]

# %% [markdown]
# # Tokenization

# %%
text_data = [pad_punctuation(x) for x in filtered_data]
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1000)

# %% [markdown]
# The last command grouped all `len(text_data)` recipes into `len(text_data)/32 == len(text_ds)` groups,
# then shuffle these groups. See [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for examples.

# %% Study tf.data.Dataset
from collections import Counter

recipes = [len(batch) for batch in text_ds]
Counter(recipes)  # Counter({32: 628, 15: 1})

# %%
vectorize_layer = layers.TextVectorization(
    standardize='lower',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQ_SIZE + 1,
)

vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # type: list, len(vocab): VOCAB_SIZE

# %% [markdown]
# # Creating the Training Set


# %%
def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_ds = text_ds.map(prepare_inputs)

# %% [markdown]
# # The Embedding Layer
# > An *embedding* layer is essentially a lookup table that converts each
# > integer token into a vector of length embedding_size, as shown in Figure 5-2.
#
# > We embed each integer token into a continuous vector because it enables the model to learn a representation for each word that is able to be updated through backpropa‐ gation.

# %%
inputs = layers.Input(shape=(None, ), dtype='int32')
x = layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)(inputs)

# %% [markdown]
# # Training the LSTM

# %%
x = layers.LSTM(UNIT_SIZE, return_sequences=True)(x)
outputs = layers.Dense(VOCAB_SIZE, activation='softmax')(x)
lstm = models.Model(inputs, outputs)
lstm.summary()

loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile('adam', loss_fn)


class TextGenerator(callbacks.Callback):

    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index
            for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature):
        prb = probs**(1 / temperature)
        prb = prb / np.sum(prb)
        return np.random.choice(len(prb), p=prb), prb

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({'prompt': start_prompt, 'word_probs': probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + ' ' + self.index_to_word[sample_token]
        print(f'\ngenerated text:\n{start_prompt}\n')
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate('recipe for', max_tokens=100, temperature=1.0)


model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint/chapter5-lstm.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

text_generator = TextGenerator(vocab)

lstm.fit(train_ds,
         epochs=25,
         callbacks=[
             model_checkpoint_callback, tensorboard_callback, text_generator
         ])

lstm.save('./models/lstm.keras')

# %% [markdown]
# # Generate Text using the LSTM model


# %%
def print_probs(info, vocab, top_k=5):
    for i in info:
        print(f"\nPrompt: {i['prompt']}")
        word_probs = i['word_probs']
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:    {np.round(100 * p, 2)}%")
        print("---------\n")


info = text_generator.generate("recipe for roasted vegetables | chop 1 /",
                               max_tokens=10,
                               temperature=1.0)
print_probs(info, vocab)

info = text_generator.generate("recipe for roasted vegetables | chop 1 /",
                               max_tokens=10,
                               temperature=0.2)
print_probs(info, vocab)

info = text_generator.generate("recipe for roasted vegetables | chop 1 /",
                               max_tokens=7,
                               temperature=1.0)
print_probs(info, vocab)

# %% test for long texts (50 tokens) generation
info = text_generator.generate("recipe for roasted vegetables | chop 1 /",
                               max_tokens=50,
                               temperature=0.2)
print_probs(info, vocab)

# %% [markdown]
# > Prompt: recipe for roasted vegetables | chop 1 / 2 cup warm water and add to a bowl .
# > add the onion , garlic , and garlic ; cook , stirring , until fragrant ,
# > about 2 minutes . add the garlic , garlic , and salt and cook , stirring
#
# The result showed that the quality of the generated texts are not good enough.

# %% test for new prompt generation
info = text_generator.generate("recipe for Japanese Sushi",
                               max_tokens=20,
                               temperature=0.2)
print_probs(info, vocab)
# %% [markdown]
# > Prompt: recipe for Japanese Sushi | combine all ingredients in a mixing glass and
# > stir well . strain into a cocktail
#
# The result showed that the narrative flow (rather than individual word)
# generated by the model are limited in the scope of the training set.
