# VQ VAE

📜 [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937)

> In this paper, we propose a simple yet powerful generative model that learns […] discrete representations [without supervision].

> Our model, the Vector Quantized Variational Auto Encoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static.

> Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes, providing further evidence of the utility of the learnt representations.

> Our goal is to achieve a model that conserves the important features of the data in its latent space while optimizing for maximum likelihood.

> In this paper, we argue for learning discrete and useful latent variables, which we demonstrate on a variety of domains.

> Learning representations with continuous features have been the focus of many previous work, however we concentrate on discrete representations which are potentially a more natural fit for many of the modalities we are interested in.

They observe that many data distributions including text, and even images given that they can be described by text, can be represented in discrete ways.

> Our model, which relies on vector quantization (VQ), is simple to train, does not suffer from large variance, and avoids the “posterior collapse” issue which has been problematic with many VAE models that have a powerful decoder, often caused by latents being ignored.

### VQ-VAE

> VAEs consist of the following parts: an encoder network which parameterizes a posterior distribution $q(z|x)$ of discrete latent random variables $z$ given the input data $x$, a prior distribution $p(z)$, and a decoder with a distribution $p(x|z)$ over input data.

**1. Discrete Latent Variables**

> We define a latent embedding space $e \in R^{K \times D}$ where $K$ is the size of the discrete latent space, and $D$ is the dimensionality of each latent embedding vector $e_i$

This model uses an encoder producing an output $z_e(x)$ for each input $x$.

Then, this value $z_e(x)$ is passed through a posterior categoorical distribution that collapses the output value into 1-of-K embedding vectors

```math
q(z = k|x) = \begin{cases}
  1 & \textrm{for } k = \textrm{argmin}_j || z_e(x) - e_j ||_2 \\
  0 & \textrm{otherwise}
\end{cases}
```

> The representation $z_e(x)$ is passed through a discretisation bottleneck followed by mapping onto the nearest element of embedding $e$.

**2. Learning**

Since $q(z)$ has no gradient, we just copy over the gradient from decoder input $z_q(x)$ to encoder output $z_e(x)$.

> Due to the straight-through gradient estimation of mapping from $z_e(x)$ to $z_q(x)$, the embeddings $e_i$ receive no gradients from the reconstruction loss $\log p(z|z_q(x))$. Therefore, in order to learn the embedding space, we use one of the simplest dictionary learning algorithms, Vector Quantization (VQ).

> The VQ objective uses the $l_2$ error to move the embedding vectors $e_i$ towards the encoder outputs $z_e(x)$.

The actual embedding space is disconnected from the main gradient flow directly, and instead is just built to minimize the overall distance between the embedding vectors and the actual encoder outputs (maximizing the utility of each embedding to matching a variety of the outputs).

> To make sure the encoder commits to an embedding and its output does not grow, we add a commitment loss, the third term in the [loss equation]. Thus, the total training objective becomes:

```math
L = \log p(x|z_q(x)) + || \textrm{sg}[z_e(x)] - e ||_2^2 | - \beta || z_e(x) - sg[e] ||_2^2
```

> where _sg_ stands for the stop-gradient operator that is defined as identity at forward computation time and has zero partial derivatives, thus effectively constraining its operand to be a non-updated constant.

The first term (familiar from VAEs) tries to maximize the probability that $x$ is regenerated given the latents (embeddings) created by $z_q(x) = q(z_e(x))$ from the encoder and categorization.

The second term minimizes the $L_2$ distance between the embedding vectors and the encoder outputs.

The third term ensures that the encoder commits to its choice of embeddings and moves its encoded outputs closer to them so that the encoder outputs don’t slowly start to diverge from the embedding choices.

**3. Prior**

> The prior distribution over the discrete latents $p(z)$ is a categorical distribution, and can be mad autoregressive by depending on other $z$ in the feature map. Whilst training the VQ-VAE, the prior is kept constant and uniform.

> After training, we fit an autoregressive distribution over $z$, $p(z)$, so that we can generate $x$ via ancestral sampling.

### Experiments

**1. Comparison with Continuous Variables**

> Our model is the first among those using discrete latent variables which challenges the performance of continuous VAEs.

**2. Images**

> Images contain a lot of redundant information as most of the pixels are correlated and noisy, therefore learning models at the pixel level could be wasteful.

> In this experiment we show that we can model $x = 128 \times 128 \times 3$ images by compressing them to a $z = 32 \times 32 \times 1$ discrete space (with $K=512$) via a purely de-convolutional $p(x|z)$.

> We model images by learning a powerful prior (PixelCNN) over $z$.

![Screenshot 2024-05-18 at 12.46.00 PM.png](../../images/Screenshot_2024-05-18_at_12.46.00_PM.png)

![Screenshot 2024-05-18 at 12.46.35 PM.png](../../images/Screenshot_2024-05-18_at_12.46.35_PM.png)

**3. Audio**

> In all our audio experiments, we train a VQ-VAE that has a dilated convolutional architecture similar to WaveNet decoder.

> This means that the VQ-VAE has, without any form of linguistic supervision, learned a high-level abstract space that is invariant to low-level features and only encodes the content of the speech.

**4. Video**

> It can be seen that the model has learnt to successfully generate a sequence of frames conditioned on given action without any degradation in the visual quality whilst keeping the local geometry correct.

![Screenshot 2024-05-18 at 12.51.34 PM.png](../../images/Screenshot_2024-05-18_at_12.51.34_PM.png)

### Conclusion

> In this work we have introduced VQ-VAE, a new family of models that combine VAEs with vector quantization to obtain a discrete latent representation.

> We have shown that VQ-VAEs are capable of modeling very long term dependencies through their compressed discrete latent space which we have demonstrated by generating 128 × 128 color images, sampling action conditional video sequences and finally using audio where even an unconditional model can generate surprisingly meaningful chunks of speech and doing speaker conversion.

> All these experiments demonstrated that the discrete latent space learnt by VQ-VAEs capture important features of the data in a completely unsupervised manner.
