# deep-learning [WIP]

**Digital intelligence has been improving at a staggering rate.**

With constant innovation at the frontiers, few have had the time to dive deep and fully understand what’s going on. On top of that, public narratives obscure the reality of the forces at play.

_This project is an effort to surface and spread the true big picture of how things are progressing, informed by the technical details of individual innovations._

**It focuses on two simple questions:**

1. How did we get here?
2. Where are we going?

In order to answer them, we first dive into the history of deep learning, understanding the entire arrow of progress from first principles. Then, using what we learn from this, we can zoom-out to observe broader trends and understand the direction we are headed in.

## Table of Contents

- [Structure](#structure)
- [Part 1: The Lessons of History](#part-1-the-lessons-of-history)
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
- [Part 2: The Arrow of Progress](#part-2-the-arrow-of-progress)
  - [1. Deep Neural Networks](/01-deep-neural-networks/) - (1) [DNN](/01-deep-neural-networks/01-dnn/), (2) [CNN](/01-deep-neural-networks/02-cnn/), (3) [AlexNet](/01-deep-neural-networks/03-alex-net/), (4) [UNet](/01-deep-neural-networks/04-u-net/)
  - [2. Optimization & Regularization](/02-optimization-and-regularization/)
    - [2.1. Weight Decay](/02-optimization-and-regularization/01-weight-decay/)
    - [2.2. ReLU](/02-optimization-and-regularization/02-relu/)
    - [2.3. Residuals](/02-optimization-and-regularization/03-residuals/)
    - [2.4. Dropout](/02-optimization-and-regularization/04-dropout/)
    - [2.5. Batch Normalization](/02-optimization-and-regularization/05-batch-norm/)
    - [2.6. Layer Normalization](/02-optimization-and-regularization/06-layer-norm/)
    - [2.7. GELU](/02-optimization-and-regularization/07-gelu/)
    - [2.8. Adam](/02-optimization-and-regularization/08-adam/)
  - [3. Sequence Modeling](/03-sequence-modeling/)
    - [3.1. RNN](/03-sequence-modeling/01-rnn/)
    - [3.2. LSTM](/03-sequence-modeling/02-lstm/)
    - [3.3. Learning to Forget](/03-sequence-modeling/03-learning-to-forget/)
    - [3.4. Word2Vec](/03-sequence-modeling/04-word2vec/)
    - [3.5. Encoder-Decoder](/03-sequence-modeling/05-encoder-decoder/)
    - [3.6. Seq2Seq](/03-sequence-modeling/06-seq2seq/)
    - [3.7. Attention](/03-sequence-modeling/07-attention/)
    - [3.8. Mixture of Experts](/03-sequence-modeling/08-mixture-of-experts/)
  - [4. Transformers](/04-transformers/)
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
  - [5. Image Generation](/05-image-generation/)
    - [5.1. GANs](/05-image-generation/01-gan/)
    - [5.2. VAEs](/05-image-generation/02-vae/)
    - [5.3. Diffusion](/05-image-generation/03-diffusion/)
    - [5.4. Stable Diffusion, ControlNet, & SDXL](/05-image-generation/04-stable-diffusion/)
    - [5.5. CLIP](/05-image-generation/05-clip/)
    - [5.6. DALL E & DALL E 2](/05-image-generation/06-dall-e/)

## Structure

This project is organized into two distinct parts (in reverse order for the sake of readability).

<br />

The first part (the remainder of this README), is a reframing of the entire history of deep learning through the lens of the fundamental constraints that have always governed the rate of progress.

These constraints are not initially obvious, but they add a new depth to what we can learn from studying the history.

Through this lens, we can explore these questions:

- How is progress made in deep learning?
- Where do the ideas that drive progress in deep learning come from?
- How have our narratives about digital intelligence changed over time?
- How does this change our understanding of intelligence, and of ourselves?
- Where are we headed, and how far off is AGI?

This part is designed to be readable for everyone, and I’ve made an effort to reference important research, my implementations, and other relevant links (tweets, videos, news, etc.) while keeping it interesting.

<br />

The second part (contained within the folders of the repository), is a technical exploration of the individual advancements that led us to the current state of digital intelligence.

A copy of each foundational paper is included in the repository, along with a detailed breakdown of the high-level ideas, mathematical intuitions, code, and relevant societal & cultural impacts for each one.

I’ve selectively included only the most essential breakthroughs that have both impacted the field, and contribute important intuitions to the broader understanding of deep learning covered in part 1.

Of course, many great papers have been left out, but the current list covers most of the important surface area necessary to understand the state of the art.

<br />

My intention is for everyone to be able to start with the broader reframing of deep learning in part 1 (below) and get value from it. Then, for the curious people who want to develop the technical intuitions, they can dive into part 2 to go deeper.

This is also why the lessons come before the actual

# Part 1: The Lessons of History

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

# Part 2: The Arrow of Progress
