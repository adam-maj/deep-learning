# deep-learning [WIP]

An exploration of the entire history of deep learning from DNNs to GPT-4o through the lens of the constraints limiting progress.

In the repository, I've included a copy of each critical paper (40+), along with my full explanations of key intuitions/math and a minimal implementation in pytorch for each topic.

On the rest of this page, I provide an overview of everything we can learn from this history, inspired by [_The Lessons of History_](https://www.amazon.com/Lessons-History-Will-Durant/dp/143914995X) by Will & Ariel Durant.

**The project is designed so everyone can get most of the value by reading the overview below.**

Then, people curious to dive into the technical details can explore the rest of the repository via the links in the [resources](#resources) section.

## Table of Contents

- [Overview](#overview-the-lessons-of-history)
  - [1. Introduction](#1-introduction)
  - [2. Constraints](#2-constraints)
    - [2.1. Data](#21-data)
    - [2.2. Size](#22-size)
    - [2.3. Optimization & Regularization](#23-optimization--regularization)
    - [2.4. Architecture](#24-architecture)
    - [2.5. Compute](#25-compute)
    - [2.6. Compute Efficiency](#26-compute-efficiency)
    - [2.7. Energy](#27-energy)
    - [2.8. Constraints & Leverage](#28-constraints--leverage)
  - [3. Narratives](#3-narratives)
  - [4. Inspiration](#4-inspiration)
  - [5. Intelligence](#5-intelligence)
  - [6. Future](#6-future)
- [Resources](#resources)
- [Papers](#papers)

# Overview

_Format inspired by [The Lessons of History](https://www.amazon.com/Lessons-History-Will-Durant/dp/143914995X) by Will and Ariel Durant_

## 1. Introduction

## 2. Constraints

### 2.1. Data

### 2.2. Size

### 2.3. Optimization & Regularization

### 2.4. Architecture

### 2.5. Compute

### 2.6. Compute Efficiency

### 2.7. Energy

### 2.8. Constraints & Leverage

## 3. Narratives

## 4. Inspiration

## 5. Intelligence

## 6. Future

# Resources

> [!IMPORTANT]
>
> Each technical breakthroughs highlighted in this repository is covered in a folder linked below.
>
> In each folder, you'll find a copy of the relevant papers (`.pdf` files), along with my own breakdown of intuitions, math, impact, and my implementation when relevant (all in the `.ipynb` file).

**1. Deep Neural Networks**

- [1.1. DNN](/01-deep-neural-networks/01-dnn/)
- [1.2. CNN](/01-deep-neural-networks/02-cnn/)
- [1.3. AlexNet](/01-deep-neural-networks/03-alex-net/)
- [1.4. UNet](/01-deep-neural-networks/04-u-net/)

**2. Optimization & Regularization**

- [2.1. Weight Decay](/02-optimization-and-regularization/01-weight-decay/)
- [2.2. ReLU](/02-optimization-and-regularization/02-relu/)
- [2.3. Residuals](/02-optimization-and-regularization/03-residuals/)
- [2.4. Dropout](/02-optimization-and-regularization/04-dropout/)
- [2.5. Batch Normalization](/02-optimization-and-regularization/05-batch-norm/)
- [2.6. Layer Normalization](/02-optimization-and-regularization/06-layer-norm/)
- [2.7. GELU](/02-optimization-and-regularization/07-gelu/)
- [2.8. Adam](/02-optimization-and-regularization/08-adam/)

**3. Sequence Modeling**

- [3.1. RNN](/03-sequence-modeling/01-rnn/)
- [3.2. LSTM](/03-sequence-modeling/02-lstm/)
- [3.3. Learning to Forget](/03-sequence-modeling/03-learning-to-forget/)
- [3.4. Word2Vec](/03-sequence-modeling/04-word2vec/)
- [3.5. Encoder-Decoder](/03-sequence-modeling/05-encoder-decoder/)
- [3.6. Seq2Seq](/03-sequence-modeling/06-seq2seq/)
- [3.7. Attention](/03-sequence-modeling/07-attention/)
- [3.8. Mixture of Experts](/03-sequence-modeling/08-mixture-of-experts/)

**4. Transformers**

- [4.1. Transformer](/04-transformers/01-transformer/)
- [4.2. BERT](/04-transformers/02-bert/)
- [4.3. RoBERTa](/04-transformers/03-roberta/)
- [4.4. T5](/04-transformers/04-t5/)
- [4.5. GPT-2](/04-transformers/05-gpt-2/)
- [4.6. GPT-3](/04-transformers/06-gpt-3/)
- [4.7. LoRA](/04-transformers/07-lora/)
- [4.8. InstructGPT](/04-transformers/08-instruct-gpt/)
- [4.9. Vision Transformer](/04-transformers/09-vision-transformer/)
- [4.10. Mixture of Experts Transformer](/04-transformers/10-moe-transformer/)

**5. Image Generation**

- [5.1. GANs](/05-image-generation/01-gan/)
- [5.2. VAEs](/05-image-generation/02-vae/)
- [5.3. Diffusion](/05-image-generation/03-diffusion/)
- [5.4. Stable Diffusion, ControlNet, & SDXL](/05-image-generation/04-stable-diffusion/)
- [5.5. CLIP](/05-image-generation/05-clip/)
- [5.6. DALL E & DALL E 2](/05-image-generation/06-dall-e/)

# Papers

Below are the citations and official links to the paper highlighted in this repository.

**Deep Neural Networks**

- **DNN** - Learning Internal Representations by Error Propagation (1987), D. E. Rumelhart et al. [[PDF]](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf)
- **CNN** - Backpropagation Applied to Handwritten Zip Code Recognition (1989), Y. Lecun et al. [[PDF]](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
- **LeNet** - Gradient-Based Learning Applied to Document Recognition (1998), Y. Lecun et al. [[PDF]](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **AlexNet** - ImageNet Classification with Deep Convolutional Networks (2012), A. Krizhevsky et al. [[PDF]](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015), O. Ronneberger et al. [[PDF]](https://arxiv.org/abs/1505.04597)

**Optimization & Regularization**

- **Weight Decay** - A Simple Weight Decay Can Improve Generalization (1991), A. Krogh and J. Hertz [[PDF]](https://proceedings.neurips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf)
- **ReLU** - Deep Learning using Rectified Linear Units (ReLU) (2018), A. Agarap [[PDF]](https://arxiv.org/pdf/1803.08375)
- **Residuals** - Deep Residual Learning for Image Recognition (2015), K. He et al. [[PDF]](https://arxiv.org/pdf/1512.03385)
- **Dropout** - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014), N. Strivastava et al. [[PDF]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- **BatchNorm** - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Ioffe and C. Szegedy [[PDF]](https://arxiv.org/pdf/1502.03167)
- **LayerNorm** - Layer Normalization (2016), J. Lei Ba et al. [[PDF]](https://arxiv.org/pdf/1607.06450)
- **GELU** - Gaussian Error Linear Units (GELUs) (2016), D. Hendrycks and K. Gimpel [[PDF]](https://arxiv.org/pdf/1606.08415)
- **Adam** - Adam: A Method for Stochastic Optimization (2014), D. P. Kingma and J. Ba [[PDF]](https://arxiv.org/pdf/1412.6980)

**Sequence Modeling**

- **RNN** - A Learning Algorithm for Continually Running Fully Recurrent Neural Networks (1989), R. J. Williams [[PDF]](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)
- **LSTM** - Long-Short Term Memory (1997), S. Hochreiter and J. Schmidhuber [[PDF]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **Learning to Forget** - Learning to Forget: Continual Prediction with LSTM (2000), F. A. Gers et al. [[PDF]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e10f98b86797ebf6c8caea6f54cacbc5a50e8b34)
- **Word2Vec** - Efficient Estimation of Word Representations in Vector Space (2013), T. Mikolov et al. [[PDF]](https://arxiv.org/pdf/1301.3781)
- **Encoder-Decoder** - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014), K. Cho et al. [[PDF]](https://arxiv.org/pdf/1406.1078)
- **Seq2Seq** - Sequence to Sequence Learning with Neural Networks (2014), I. Sutskever et al. [[PDF]](https://arxiv.org/pdf/1409.3215)
- **Attention** - Neural Machine Translation by Jointly Learning to Align and Translate (2014), D. Bahdanau et al. [[PDF]](https://arxiv.org/pdf/1409.0473)
- **Mixture of Experts** - Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017), N. Shazeer et al. [[PDF]](https://arxiv.org/pdf/1701.06538)

**Transformers**

- **Transformer** - Attention Is All You Need (2017), A. Vaswani et al. [[PDF]](https://arxiv.org/pdf/1706.03762)
- **BERT** - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018), J. Devlin et al. [[PDF]](https://arxiv.org/pdf/1810.04805)
- **RoBERTa** - RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019), Y. Liu et al. [[PDF]](https://arxiv.org/pdf/1907.11692)
- **T5** - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019), C. Raffel et al. [[PDF]](https://arxiv.org/pdf/1910.10683)
- **GPT-2** - Language Models are Unsupervised Multitask Learners (2018), A. Radford et al. [[PDF]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **GPT-3** - Language Models are Few-Shot Learners (2020) T. B. Brown et al. [[PDF]](https://arxiv.org/pdf/2005.14165)
- **LoRA -** LoRA: Low-Rank Adaptation of Large Language Models (2021), E. J. Hu et al. [[PDF]](https://arxiv.org/pdf/2106.09685)
- **InstructGPT** - Training language models to follow instructions with human feedback (2022), L. Ouyang et al. [[PDF]](https://arxiv.org/pdf/2203.02155)
- **Vision Transformer** - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020), A. Dosovitskiy et al. [[PDF]](https://arxiv.org/pdf/2010.11929)
- **Mixture of Experts Transformer** - Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models (2023), S. Shen et al. [[PDF]](https://arxiv.org/pdf/2305.14705)

**Generative Models**

- **GAN** - Generative Adversarial Networks (2014), I. J. Goodfellow et al. [[PDF]](https://arxiv.org/pdf/1406.2661)
- **Style GAN** - A Style-Based Generator Architecture for Generative Adversarial Networks (2018), T. Karras et al. [[PDF]](https://arxiv.org/pdf/1812.04948)
- **Style GAN 2** - Analyzing and Improving the Image Quality of StyleGAN (2019), T. Karras et al. [[PDF]](https://arxiv.org/pdf/1912.04958)
- **VAE** - Auto-Encoding Variational Bayes (2013), D. Kingma and M. Welling [[PDF]](https://arxiv.org/pdf/1312.6114)
- **VQ VAE** - Neural Discrete Representation Learning (2017), A. Oord et al. [[PDF]](https://arxiv.org/pdf/1711.00937)
- **VQ VAE 2** - Generating Diverse High-Fidelity Images with VQ-VAE-2 (2019), A. Razavi et al. [[PDF]](https://arxiv.org/pdf/1906.00446)
- **Diffusion** - Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015), J. Sohl-Dickstein et al. [[PDF]](https://arxiv.org/pdf/1503.03585)
- **Denoising Diffusion** - Denoising Diffusion Probabilistic Models (2020), J. Ho. et al. [[PDF]](https://arxiv.org/pdf/2006.11239)
- **Denoising Diffusion 2** - Improved Denoising Diffusion Probabilistic Models (2021), A. Nichol and P. Dhariwal [[PDF]](https://arxiv.org/pdf/2102.09672)
- **Diffusion Beats GANs** - Diffusion Models Beat GANs on Image Synthesis, P. Dhariwal and A. Nichol [[PDF]](https://arxiv.org/pdf/2105.05233)
- **Stable Diffusion** - High-Resolution Image Synthesis with Latent Diffusion Models (2021), R. Rombach et al. [[PDF]](https://arxiv.org/pdf/2112.10752)
- **ControlNet** - Adding Conditional Control to Text-to-Image Diffusion Models (2023), L. Zhang et al. [[PDF]](https://arxiv.org/pdf/2302.05543)
- **SDXL** - SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis (2023), D. Podell et al. [[PDF]](https://arxiv.org/pdf/2307.01952)
- **CLIP** - Learning Transferable Visual Models From Natural Language Supervision (2021), A. Radford et al. [[PDF]](https://arxiv.org/pdf/2103.00020)
- **DALL E** - Zero-Shot Text-to-Image Generation (2021), A. Ramesh et al. [[PDF]](https://arxiv.org/pdf/2102.12092)
- **DALL E 2** - Hierarchical Text-Conditional Image Generation with CLIP Latents (2022), A. Ramesh et al. [[PDF]](https://arxiv.org/pdf/2204.06125)
