# GAN

📜 [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1)

> We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates the probability that a sample came from the training data rather than $G$.

> This framework corresponds to a minimax two-player game.

> Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context.

### Adversarial Nets

> We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train $G$ to minimize $\log(1 − D(G(z)))$

> In other words, $D$ and $G$ play the following two-player minimax game with value function $V(G,D)$

```math
\underset{G}{\textrm{min}} \, \underset{D}{\textrm{max}} \,
V(D,G) = \mathbb{E}_{x \sim p_{\textrm{data}}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_x(z)} [\log(1 - D(G(z))]
```

> Rather than training G to minimize $\log(1 − D(G(z)))$ we can train $G$ to maximize $\log D(G(z))$. This objective function results in the
> same fixed point of the dynamics of $G$ and $D$ but provides much stronger gradients early in learning.

Allowing the generator to maximize the probability of fooling the discriminator, rather than minimize the probability of getting discovered by the discriminator, provides a better optimization since it’s very easy for the discriminator to vote against the generator early on.

### Experiments

![Screenshot 2024-05-18 at 2.04.41 PM.png](../../images/Screenshot_2024-05-18_at_2.04.41_PM.png)

### Advantages and Disadvantages

> The disadvantages are primarily that there is no explicit representation of $p_g(x)$, and that $D$ must be synchronized well with $G$ during training (in particular, G must not be trained too much without updating D).

> The advantages are that Markov chains are never needed, only back-prop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model.

Generative adversarial models are far more computationally efficient than the previous Markov chain based models.
