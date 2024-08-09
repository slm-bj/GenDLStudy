# %% [markdown]
# # Generate Texts with a Pretained LSTM Model
#
# This script shows how to generate texts with a pretrained LSTM model.
# It's written based on script lstm.py in the same folder.

# %%
import re
import sys
import json
import string
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models

VOCAB_SIZE = 10000
SEQ_SIZE = 200  # number of tokens (words) in a sequence
MODEL_PATH = "./models/lstm.keras"

# %% [markdown]
# # Load Data and Generate Vocabulary

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


with open('../data/full_format_recipes.json') as jd:
    recipe_data = json.load(jd)

filtered_data = [
    f"Recipe for {x['title']} | {' '.join(x['directions'])}"
    for x in recipe_data if 'title' in x and x['title'] is not None
    and 'directions' in x and x['directions'] is not None
]

text_data = [pad_punctuation(x) for x in filtered_data]
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1000)

vectorize_layer = layers.TextVectorization(
    standardize='lower',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQ_SIZE + 1,
)

vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # type: list, len(vocab): VOCAB_SIZE

# %% [markdown]
# # Load Model and Build Text Generator

# %%
if not Path(MODEL_PATH).exists():
    sys.exit(f"Model file {MODEL_PATH} does not exist!")

lstm = models.load_model(MODEL_PATH, compile=False)

class TextGenerator:

    def __init__(self, index_to_word, model, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index
            for index, word in enumerate(index_to_word)
        }
        self.model = model

    def _sample_from(self, probs, temperature):
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
            sample_token, probs = self._sample_from(y[0][-1], temperature)
            info.append({'prompt': start_prompt, 'word_probs': probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + ' ' + self.index_to_word[sample_token]
        print(f'\ngenerated text:\n{start_prompt}\n')
        return info


text_generator = TextGenerator(vocab, lstm)

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
# > Prompt: recipe for roasted vegetables | chop 1 / 2 cup
# > warm water and add to the bowl of water . bring to a boil , then
# > reduce heat to medium - low and cook , stirring occasionally ,
# > until vegetables are tender , about 10 minutes . add the
# The generated texts are legible!

# %% test for new prompt generation
info = text_generator.generate("recipe for Japanese Sushi",
                               max_tokens=30,
                               temperature=0.2)
print_probs(info, vocab)
# %% [markdown]
# > Prompt: recipe for Japanese Sushi | combine all ingredients in a cocktail shaker
# > and shake vigorously . strain into a cocktail glass .

