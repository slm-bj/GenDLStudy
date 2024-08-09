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
