# Chapter 1

## What Is Generative Modeling?

### Generative vs Discriminative Modeling

> Generative modeling is a branch of machine learning that involves training a model to produce new data that is similar to a given dataset.

Key concetps (Figure 1-1):

* Training data: all horse images;
* Observation: sample, one image;
* Feature: a pixel;
* Probabilistic (generative): mimic the distribution as close as posible, then sample from it to generate a new one;
* Deterministic (discriminative): e.g.: Figure 1-2, if a painting is from Van Gogh?
* Label: attached to each observation. Question: does generative model have a label?

Generative vs discriminative modeling:
estimate $p(x)$ or $p(x|y)$ (conditional generative, e.g.: generate apple in a fruit dataset)
    vs $p(y|x)$.

### The Rise of Generative Modeling

Why we know more about discriminative than generative learning?

* Discriminative learning is easier: but genAI is in fast evolution, Figure 1-3;
* More applicable in industry: but genAI is becoming more and more practical, for example ChatGPT, Sora, etc;

### Generative Modeling and AI

Three relations between generative modeling and AI:

* We should understand the distribution of the data. Compared with clustering in unsupervised learning;
* A good tool used in other AI scenario, e.g.: reinforcement learning with world model in Chapter 12;
* A road to AGI;

## Our First Generative Model

### Hello World for GenAI

Question: guess Figure 1-4.

Figure 1-5.

### The Generative Modeling Framework

The Generative Modeling Framework: from $p_{data}$ to $p_{model}$.

Discussion: in Figure 1-6, besides rectangle distribution, other solutions are also reasonable.

### Representation Learning

We human recognize people with high level *representation* (e.g.: gender, height, hair color, eye color)
rather than low level feature (e.g.: a pixel in the image).

> Each point in the low dimension *latent space* is a *representation* of some high-dimensional observation.

Introduce *representation learning* with an example Figure 1-7. 

*Encoder*: High dimensional *manifold* (high dimensional plane in pixel space) -> low dimensional latent space;

*Decoder*: from latent space back to pixel space;

## Core Probability Theory

Relationship between deep learning and **statistical** modeling of probability of distribution.

Core concepts:

*Sample space*: the complete set of all values an observation *can* take. In Figure 1-4, it includes all continents and seas.

*Probability density function*: add up to 1.

> there are *infinitely* many density functions $p_{model}(x)$ that we can use to estimate $p_{data}(x)$.

*Parametric modeling*: a *parametric model* is a family of density functions $p_{\theta}(x)$
that can be described using a finite number of parameters, $\theta$.

*Likelihood*:

Question: why "in the world map example, an orange box that only covered the left half of the map would have a likelihood of 0"?

Because under that $p_{\theta}$, samples in the right half of the map have $p(x) = 0$.
Therefore the multiplication of all $p(x)$ is 0.

> The likelihood of a set of parameters $\theta$ is defined to be the probability of
> seeing the data if the true data-generating distribution was the model parameterized by $\theta$.

> the likelihood is a function of the *parameters*, not the *data*.

*Maximum likelihood estimation*: MLE.

Question: why not make a very small model space which only contains those dot samples?

Answer: this model space has much more parameters to describe (many isolated small circles)
than a simple rectangle (which contains only 4 parameters).

## Generative Model Taxonomy

> The first split that we can make is between models where the probability density function $p(x)$
> is modeled explicitly and those where it is modeled implicitly.

* Explicit density: we convert PDF to some simpler and computable form to get MLE.
* Implicit density: generate data directly without the knowledge of probability density.

In explicit density method, to make PDF computable, we have 2 ways:

* Tractable model: the PDF of the model is simple enough to compute;
* Approximate density model: we use approximation of PDF to get MLE.

## Codebase

```sh
git clone git@github.com:slm-bj/GenDLStudy.git
```
