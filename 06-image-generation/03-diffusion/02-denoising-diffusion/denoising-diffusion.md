# Denoising Diffusion

📜 [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

> We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from non-equilibrium thermodynamics.

> Our best results are obtained by training on a weighted variational
> bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.

> And our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding.

> We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models.

> We find that the majority of our models’ lossless code-lengths are consumed to describe imperceptible image details.

> We show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models.

### Background

…

### Diffusion Models and Denoising Autoencoders

> To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching that leads to a simplified, weighted variational bound objective for diffusion models.

**1. Forward Process and $L_T$**

> In our implementation, the approximate posterior $q$ has no learnable parameters, so $L_T$ is a constant during training and can be ignored.

…

### Conclusion

> We have presented high quality image samples using diffusion models, and we have found connections among diffusion models and variational inference for training Markov chains, denoising score matching and annealed Langevin dynamics (and energy-based models by extension), autoregressive models, and progressive lossy compression.
