# DALL E 2

📜 [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/pdf/2204.06125)

> We propose a two-stage model: a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding.

> Our decoders conditioned on image representations can also produce variations of an image that preserve both its semantics and style, while varying the non-essential details absent from the image representation.

> Moreover, the joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion.

The creation of CLIP has enabled far more robust text-to-image models by adding an image decoder that can convert from CLIP embeddings to an image.

> We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior, finding that the latter are computationally more efficient and produce higher-quality samples.

Using diffusion models for both parts of the model appears to be the best approach.

> In this work, we combine these two approaches [CLIP and diffusion models] for the problem of text-conditional image generation. We first train a diffusion _decoder_ to invert the CLIP image _encoder_.

> One notable advantage of using the CLIP latent space (over GANs) is the ability to semantically modify images by moving in the direction of any encoded text vector, whereas discovering these directions in GAN latent space involves luck and diligent manual examination.

Because of the syntactically and semantically consistent embeddings of the CLIP latent space, manipulating images is possible using text, whereas this is intractable with GANs.

> To obtain a full generative model of images, we combine the CLIP image embedding decoder with a prior model, which generates possible CLIP image embeddings from a given text caption.

The prior model is meant to enhance the CLIP image embeddings from the original text caption to make them more conducive to generating good images (which may mean enriching them with more description, etc.)

### Method

We can model the combined action of the _prior_ and _decoder_ as follows

```math
P(x|y) = P(x, z_i|y) = P(x|z_i, y)P(z_i|y)
```

Here, we model the distribution of the probability of an image $x$ given the caption $y$ by splitting it into the prior, which models the probability of a given image embedding $z_i$ given the caption $y$, and then the probability of an image $x$ given the image embedding $z_i$, and optionally, the caption $y$ as well.

**1. Decoder**

> We use diffusion models to produce images conditioned on CLIP image embeddings.

> Specifically, we modify the architecture […] by projecting and adding CLIP embeddings to the existing time-step embedding, and by projecting CLIP embeddings into four extra tokens of context that are concatenated to the sequence of outputs from the GLIDE text encoder.

**2. Prior**

> For the diffusion prior, we train a decoder-only Transformer with a causal attention mask on a sequence consisting of, in order: the encoded text, the CLIP text embedding, an embedding for the diffusion time-step, the noised CLIP image embedding, and a final embedding whose output from the Transformer is used to predict the un-noised CLIP image embedding.

### Image Manipulations

> Our approach allows us to encode any given image $x$ into a bipartite latent representation $(z_i, x_T)$ that is sufficient for the decoder to produce an accurate reconstruction.

**1. Variations**

![Screenshot 2024-05-17 at 2.47.58 PM.png](../../images/Screenshot_2024-05-17_at_2.47.58_PM.png)

> Given an image $x$, we can produce related images that share the same essential content but vary in other aspects, such as shape and orientation.

**2. Interpolations**

![Screenshot 2024-05-17 at 2.48.14 PM.png](../../images/Screenshot_2024-05-17_at_2.48.14_PM.png)

> It is also possible to blend two images $x_1$ and $x_2$ for variations, traversing all of the concepts in CLIP’s embedding space that occur between them.

**3. Text Diffs**

![Screenshot 2024-05-17 at 2.48.28 PM.png](../../images/Screenshot_2024-05-17_at_2.48.28_PM.png)

> A key advantage of using CLIP compared to other models for image representations is that it embeds images and text to the same latent space, thus allowing us to apply language-guided image manipulations (text-diffs).
