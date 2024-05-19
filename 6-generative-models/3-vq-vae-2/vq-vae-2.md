# VQ VAE 2

> We scale and enhance the autoregressive priors used in VQ-VAE to generate synthetic samples of much higher coherence and fidelity than possible before.

> We demonstrate that a multi-scale hierarchical organization of VQ-VAE, augmented with powerful priors over the latent codes, is able to generate samples with quality that rivals that of state of the art Generative Adversarial Networks on multifaceted datasets such as ImageNet, while not suffering from GAN’s known shortcomings such as mode collapse and lack of diversity

> It is well known that samples from [GANs] do not fully capture the diversity of the true distribution.

> In contrast, likelihood based methods optimize negative log-likelihood (NLL) of the training data. This objective allows model-comparison and measuring generalization to unseen data.

> In this paper we use ideas from lossy compression to relieve the generative model from modeling negligible information.

### Background

**1. Vector Quantized Variational Auto Encoder**

> The VQ-VAE model can be better understood as a communication system. It comprises of an encoder that maps observations onto a sequence of discrete latent variables, and a decoder that reconstructs the observations from these discrete variables. Both encoder and decoder use a shared codebook.

The VQ-VAE has a _codebook_ that it uses to communicate between the encoder and the decoder.

The decoder maps the received indices of vectors in the codebook and uses it to reconstruct the original data via non-linearities. This is the “regeneration loss.”

In addition, the VQ-VAE has _codebook loss_ to make the codebook match more closely with the encoder outputs, and the _commitment loss_ to encourage the output of the decoder to stay closer to the codebook.

```math
\mathcal{L}(x, D(e)) = ||x - D(e)||_2^2 + ||sg[E(x)] - e||_2^2 + \beta || sg[e] - E(x) ||_2^2
```

**2. PixelCNN Family of Autoregressive Models**

> Deep autoregressive models are common probabilistic models that achieve state of the art results in density estimation across several data modalities.

### Method

> The proposed method follows a two-stage approach: first, we train a hierarchical VQ-VAE to encode images onto a discrete latent space, and then we fit a powerful PixelCNN prior over the discrete latent space induced by all the data.

This sets the intuition for transformer based image generation with VQ-VAEs as well - the place where PixelCNN is operating now can be replaced with a transformer with self-attention to learn the distribution.

![Screenshot 2024-05-18 at 1.28.31 PM.png](../../images/Screenshot_2024-05-18_at_1.28.31_PM.png)

**1. Stage 1: Learning Hierarchical Latent Codes**

> As opposed to vanilla VQ-VAE, in this work we use a hierarchy of vector quantized codes to model large images. The main motivation behind this is to model local information, such as texture, separately from global information such as shape and geometry of objects.

> The prior model over each level can thus be tailored to capture the specific correlations that exist in that level.

> The structure of our multi-scale hierarchical encoder [has] a top latent code which models global information, and a bottom latent code, conditioned on the top latent, responsible for representing local details.

**2. Stage 2: Learning Priors over Latent Codes**

> In order to further compress the image, and to be able to sample from the model learned during, we learn a prior over the latent codes

This separate model takes the embedding space learned by the auto-encoder and learns a prior on it.

In this way, the auto-encoder has done the job of compressing the data in a way to get rid of less important information where it’s encoded outputs only represent important data to recreate the image.

Then, when we train a neural network to learn the prior on the encoders output distribution, it’s effectively modeling the actual data generating distribution much more efficiently than if it were observing the original data since all the noise has been removed and only important features remain.

This makes learning the prior distribution a powerful way to sample other points in the original state space that actually correspond with likely values.

> From an information theoretic point of view, the process of fitting a prior to the learned posterior can be considered as lossless compression of the latent space by re-encoding the latent variables with a distribution that is a better approximation of their true distribution, and thus results in bit rates closer to Shannon’s entropy.

The auto-encoder provides compression to a new distribution that can be modeled more effectively than the original due to the removal of noise.

> In the VQ-VAE framework, this auxiliary prior is modeled with a powerful, autoregressive neural network such as PixelCNN in a post-hoc, second stage.

**3. Trading off Diversity with Classifier Based Rejection Sampling**

> Unlike GANs, probabilistic models trained with the maximum likelihood objective are forced to model all of the training data distribution.

Because of this, having samples in the dataset that don’t match nicely with the proper underlying distribution adds a considerable challenge in actually converging to the correct data distribution.

To mitigate this, they create an automated way to classify the quality of the samples.

> In this work, we also propose an automated method for trading off diversity and quality of samples based on the intuition that the closer our samples are to the true data manifold, the more likely they
> are classified to the correct class labels by a pre-trained classifier.

Using this method, they can classify the quality of the samples by their proximity to the underlying distribution.

### Experiments

> In this section, we present quantitative and qualitative results of our model trained on ImageNet 256 × 256.

![Screenshot 2024-05-18 at 1.46.40 PM.png](../../images/Screenshot_2024-05-18_at_1.46.40_PM.png)

![Screenshot 2024-05-18 at 1.48.12 PM.png](../../images/Screenshot_2024-05-18_at_1.48.12_PM.png)

**1. Modeling High-Resolution Face Images**

> Although modeling faces is generally considered less difficult compared to ImageNet, at such a high resolution there are also unique modeling challenges that can probe generative models in interesting ways. For example, the symmetries that exist in faces require models capable of capturing long range dependencies.

### Conclusion

> We propose a simple method for generating diverse high resolution images using VQ-VAE with a powerful autoregressive model as prior.

> Our encoder and decoder architectures are kept simple and light-weight as in the original VQ-VAE, with the only difference that we use a hierarchical multi-scale latent maps for increased resolution.

> We believe our experiments vindicate autoregressive modeling in the latent space as a simple and effective objective for learning large scale generative models.
