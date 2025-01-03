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
* 2024.10.20: Transformers, coding discussion, GuangYu
* 2024.10.27: ProGAN in Advanced GANs, Xiaojie
* 2024.11.3: StyleGAN & StyleGAN2, Xiaojie
* 2024.11.16: StyleGAN2, Xiaojie
* 2024.11.30: Xiaojie: VQ-GAN & Leo: Music generation
* 2024.12.8: Xiaojie: ViT VQ-GAN
* 2024.12.15: Leo: Music Transformer
* 2024.12.22: Leo: MuseGAN


## Running Codes

Install [uv](https://github.com/astral-sh/uv) and run:
```sh
uv pip install -r pyproject.toml
uv run python <script>.py  # or:
uv run ipython music_transformer.ipy
```

## Music Generation Setup

Download [󰌷 AppImage](https://musescore.org/en/download/musescore-x86_64.AppImage)
of MuseScore and run:
```sh
chmod 755 MuseScore-Studio-4.4.3.242971445-x86_64.AppImage
cp MuseScore-Studio-4.4.3.242971445-x86_64.AppImage ~/.local/bin/
uv run python -m music21.configure
```

Then change path of MuseScore in ~/.music21rc as follows:
```xml
...
  <preference name="musescoreDirectPNGPath" value="/home/leo/.local/bin/MuseScore-Studio-4.4.3.242971445-x86_64.AppImage" />
  <preference name="musicxmlPath" value="/home/leo/.local/bin/MuseScore-Studio-4.4.3.242971445-x86_64.AppImage" />
...
```

Now durations and notes can be generated by running:
```sh
uv run ipython music_transformer.ipy
```
