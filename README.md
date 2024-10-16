# GenDLStudy

Study notes of
[Generative Deep Learning, 2nd Edition](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/)
by David Foster.

## Symposia

* 2024.6.1: [Preface](Preface.md), Leo
* 2024.6.8: [Chapter 1](Chapter1.md), Leo
* 2024.6.15: [Chapter 2](Chapter2.md), Leo
* 2024.6.22: part 1 of [Chapter 3](Chapter3.md), Sharon
* 2024.7.6: VAE of chapter 3, Sharon
* 2024.7.20: DCGAN of chapter 4, GuangYu
* 2024.7.27: WGAN-GP of chapter 4, Xiaojie
* 2024.8.4: CGAN of chapter 4, Leo
* 2024.8.10: LSTM of chapter 5, Sharon
* 2024.8.18: GRU and Bidirectional LSTM of chapter 5, Sharon
* 2024.8.25: Pixel CNN, GuangYu
* 2024.9.1: chapter 6, Real NVP, Xiaojie
* 2024.9.8: Real NVP
* 2024.9.16: Energy-based Models, theory study
* 2024.9.22: Energy Based Models, implementation discussion
* 2024.9.28: Diffusion Models, theory discussion
* 2024.10.6: Diffusion Models, theory discussion continued, Sharon and I
* 2024.10.13: Transformers, theory discussion, GuangYu

## Running Codes

Some Python scripts are converted from Jupyter notebook to avoid
Base64 strings messed up version history.
These scripts can be executed in 3 ways:

First as Python script:
```sh
cd scripts
poetry run ipython <script>.py
```

Or convert to Jupyter notebook and play with it interactively:
```sh
poetry jupytext --to notebook <script.py>
```

Lastly you can convert and execute it in one go:
```sh
poetry jupytext --to notebook --execute <script.py>
```

Choose one from them according to your needs.
