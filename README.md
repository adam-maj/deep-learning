# deep-learning [WIP]

An exploration of the entire history of deep learning progress that got us from simple feed-forward networks to GPT-4o and what we can learn from it.

Covers all the essentials in depth, including:

- Deep learning basics
- CNNs
- Optimization
- Regularization
- RNNs
- LSTMs
- Attention
- Transformers
- Embeddings
- GPTs
- RLHF
- Mixture of Experts
- GANs
- Diffusion
- VQ-VAEs
- Multi-modality

Each paper is in its own section with:

1. a copy of the paper itself
2. my notes highlighting important observations & explaining key concepts
3. a fully documented minimal implementation in pytorch when relevant

Finally, I've included my observations on what this history can teach us about:

1. how progress is made in deep learning broadly
2. the constraints that govern the limit on digital intelligence
3. where the field is going

### Philosophy

**1. Minimal but complete:**

Covers all the essentials while leaving out all the noise and less important details. I've tried to curate the list of papers highlighted here to make sure that each one brings significant unique value, and have cut out everything non-essential.

**2. Technical but approachable:**

By going through papers & implementations (rather than dumbed down high-level explanations), you're forced to grapple with and fully understand the technical details at the lowest level. I've tried to make this process more approachable by including my own notes and observations.

**3. Up-to-date:**

The list contains everything you need to understand the fundamentals of all the state-of-the-art models as of 2024.

**4. Focused on building intuitions:**

In my experience, you develop much more robust intuitions by going through the history yourself, observing what worked and what didn't, how progress was made, etc. rather than just getting a dilluted version from secondary sources.

**5. Practicality first:**

I've prioritized highlighting concepts that have led to the development of significant breakthroughs in AI that are already in production. I've avoided highlighting interesting research frontiers since (1) most frontiers will soon be obselete by definition, so the time spend is not worth it in introductory learning (2) I'm not qualified to suggest research frontiers.

### Implementations

Each of the following will be implemented:

Architectures:

- DNN
- CNN
- RNN
- LSTM
- Transformer
- GAN
- VAE
- UNet
- Diffusion

Optimization:

- Adam
- Dropout
- LayerNorm
