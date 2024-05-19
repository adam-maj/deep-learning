# VAE

📜 [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)

> How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets?

How do we make probabilistic models (where parts of the feed-forward process come from sampling from a distribution, rather than all being deterministic) on large datasets where the apparent data generating distribution is continuous an intractable?

An example of this is large image datasets, where the actual distribution producing each image is extremely complex, as lots of noise is permitted and image content will still remain the same (this is an indicator of the complexity of the distribution - you have to account for so many possibilities, which are all continuous in terms of pixel values and positions).

> We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case.

**variational inference algorithm** - an algorithm that models the true (intractable) distribution of the data with a much simpler variational distribution, like the Gaussian distribution, which allows simpler inference

Importantly, modeling the true distribution with a simpler and differentiable distributions means we can actually compute gradients

> We show that a re-parameterization of the variational lower bound
> yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods.

One of the most important contributions of the paper - they provide a way to alter the random sampling component (from a Gaussian) usually present in the variational lower bound optimization (explained later) that makes the random sampling actually differentiable.

This is critical - this innovation makes it actually possible to use stochastic gradient descent and deep neural networks with back-propagation to optimize probabilistic models.

> Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator.

Another key innovation - they show that you can model the extremely complex and unknown posterior distribution with a separate internal distribution in the model, which can be tuned, and then used to run inference.

This recognition model can be optimized using the lower bound estimator.

> The variational Bayesian (VB) approach involves the optimization of an approximation to the intractable posterior.

The inspiration behind the VAE approach from statistics. The variational Bayesian approach, which models a more complex distribution with a simpler one (like the Gaussian).

> Unfortunately, the common mean-field approach requires analytical solutions of expectations w.r.t. the approximate posterior, which are also intractable in the general case.

The usual approach to VB is the “mean-field” approach which involves calculating an expectation of conditional probabilities based on random variables sampled from a distribution - and as a result often involves integrating over CDFs which takes “analytical solutions” that are often intractable.

> We show how a re-parameterization of the variational lower bound yields a simple differentiable unbiased estimator of the lower bound; this SGVB (Stochastic Gradient Variational Bayes) estimator can be used for efficient approximate posterior inference in almost any model with continuous latent variables and/or parameters, and is straightforward to optimize using standard stochastic gradient ascent techniques.

They introduce a new method that updates on this VB method to make it usable in stochastic gradient descent by addressing the problems brought up above.

> We propose the AutoEncoding VB (AEVB) algorithm. In the AEVB algorithm we make inference and learning especially efficient by using the SGVB estimator to optimize a recognition model that allows us to perform very efficient approximate posterior inference using simple ancestral sampling

They create a joint model where one part learns to form a recognition model that approximates the more complex data generating distribution, and then an inference model that samples from this distribution to reproduce the original values.

> Which in turn allows us to efficiently learn the model parameters, without the need of expensive iterative inference schemes (such as MCMC) per datapoint.

This architecture with both a recognition model and an inference model forces the recognition model to encode useful representations that can be reinterpreted back to the original data.

This method is far more efficient than previous methods that require long chains of compute like Markov Chain Monte Carlo to accomplish the same thing.

> When a neural network is used for the recognition model, we arrive at the variational auto-encoder.

### Method

> We will restrict ourselves […] to […] where we like to perform maximum likelihood (ML) or maximum a posteriori (MAP) inference on the (global) parameters, and variational inference on the latent variables.

**1. Problem Scenario**

We deal with situations where we have some dataset $X = \{ x^{(i)} \}_{i=1}^N$ that consists of a process of values where (1) some values $z^{(i)}$ are generated from some distribution $p_{\theta^*}(z)$ and (2) the values $x^{(i)}$ are drawn from $p_{\theta^*}(x|z)$, and that their PDFs are tractable, but that calculating the integral of the marginal likelihood $\int p_\theta(z) p\theta(x|z) dz$ (necessary to calculate the evidence lower bound) is intractable.

> We are interested in, and propose a solution to, three related problems in the above scenario:

1. Efficient approximate ML or MAP estimation for the parameters $\theta$. The parameters […] allow us to mimic the hidden random process and generate artifical data that resembles the real data.

2. Efficient approximate posterior inference of the latent variable $z$ given an observed value $x$ for a choice of parameters $\theta$. This is useful for coding or data representation tasks.

3. Efficient approximate marginal inference of the variable $x$. This allows us to perform all kinds of inference tasks where a prior over $x$ is required. Common applications in computer vision include image denoising, in-painting and super-resolution.
   >

Each of these 3 objectives provides real value that will become important. The representation space created by the encoder becomes very valuable (and can even be used for embeddings).

Additionally, the inference part can be used for many image generation and manipulation applications.

We introduce a “recognition model” $q_\phi(z|x)$ meant to model the true posterior distribution, which will be called the _encoder_. Then, the distribution $p_\theta(x|z)$ will be called the _decoder_.

**1. The Variational Bound**

> The marginal likelihood is composed of a sum over the marginal likelihoods of individual datapoints $\log p_\theta(x^{(1)}, …, x^{(N)}) = \sum_{i=1}^N \log p_\theta(x^{(i)})$, which can be rewritten as:

```math
\log p_\theta(x^{(i)}) = D_{KL}(q_\phi(z|x^{(i)}), p_\theta(z|x^{(i)})) + \mathcal{L}(\theta, \phi; x^{(i)})
```

The KL divergence models the difference between our approximate distribution and the true distribution of the latent variables $z$ given the actual sampled value $x^{(i)}$.

The second term $\mathcal{L}$, known as the _variation lower bound_, marks the lower bound of the probability of any observed data-point appearing in our distribution:

```math
\log p_\theta(x^{(i)}) \geq \mathcal{L}(\theta, \phi; x^{(i)}) = \mathbb{E}_{q_\phi(z|x)}[-\log q_\phi(z|x) + \log p_\theta(x,z)]
```

This term re presents the evidence lower bound. We take the expectation of values sampled over our approximate distribution $q_\phi$ for all values $z$ sampled given $x$, weighted by their relative probabilities.

For each, we want to maximize the first term (the entropy) $- \log q_\phi(z|x)$, which means minimizing the probability of $z$ being sampled given $x$ generally. This spreads out the distribution more, preventing $z$ from concentrating around too few values.

The second term $\log p_\theta(x,z)$ ensures that our model learns useful representations for $z$ that allow the decoder to actually reconstruct $x$.

These two terms work together to make the ELBO effective. The first term can be thought of as saying “explore a broad range of possibilities for $z$ for each $x$, instead of narrowing down” forcing the distribution to be very spread out (different $z$ values should not be highly used across the model).

Meanwhile, the second term is saying “focus on the values of $z$ that effectively help the decoder to recreate $x$.”

We can then rewrite the above equation for the ELBO using the KL divergence as follows:

```math
\mathcal{L}(\theta, \phi; x^{(i)}) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(z) + \log p_\theta(x|z) - \log q_\phi(z|x)] \\

D_{KL}(q_\phi(z|x), p_\theta(z)) = \mathbb{E}_{q_\phi(z|x)}[\log q_\theta(z|x) - \log p_\theta(z)] \\

\mathcal{L}(\theta, \phi; x^{(i)}) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x), p_\theta(z))
```

Thus, the first term of this maximizes the probability that our decoder can produce $x$ from the $z$ values sampled from the encoder, and the second term minimizes the difference between the approximate distributions condition $z|x$ and the overall sampling of $z$.

**3. The SGVB Estimator and AEVB Algorithm**

> In this section we introduce a practical estimator of the lower bound and its derivates w.r.t the parameters.

Here, we address the ability to model our evidence lower bound for maximization with an efficient estimator.

> Under certain mild conditions […] for a chosen approximate posterior $q_\phi (z|x)$ we can reparameterize random variable $\hat{z} \sim q_\phi(z|x)$ using a differentiable transformation $g_\phi(\epsilon, x)$ of an (auxiliary) noise variable $\epsilon$

```math
\hat{z} = g_\phi(\epsilon, x) \textrm{ with } \epsilon \sim p(\epsilon) \\

E_{q_\phi(z|x^{(i)})}[f(z)] = E_{p(\epsilon)}[f(g_\phi(\epsilon, x^{(i)}))] \simeq \frac{1}{L} \sum_{l=1}^L f(g_\phi(\epsilon^{(l)}, x^{(i)})) \\
```

Here, we create a re-parameterized estimator introducing some random noise through the variable $\epsilon$ which introduces noise. Then, you can take the expectation by taking the average over a number of samples influenced by this random noise, rather than calculating the integral through analytical means.

We can apply this sampling method to our calculation of $\mathcal{L}$ to create an estimator function $\mathcal{L}^A(\theta, \phi; x)$ that can be used to optimize the ELBO.

> The KL-divergence term can then be interpreted as regularizing $\phi```, encouraging the approximate posterior to be close to the prior $p_\theta(z)$.

> A connection with auto-encoders becomes clear when looking at the objective function. The first term is (the KL divergence of the approximate posterior from the prior) acts as a regularizer, while the second term is a an expected negative reconstruction error.

**4. The re-parameterization trick**

> The essential parameterization trick is quite simple. Let $z$ be a continuos random variable, and $z \sim q_\phi(z|x)$ be some conditional distribution. It is then often possible to express the random variable $z$ as a deterministic variable $z = g_\phi(\epsilon, x)$ where $\epsilon$ is an auxiliary variable with independent marginal $p(\epsilon)$ and $g_\phi(.)$ is some vector-valued function parameterized by $\phi$.

This reframing (re-parameterization) allows us to make the sampling of the random variable $z$ differentiable (via the parameters $\phi$) by isolating the noise to a separate variable $\epsilon$.

> Take, for example, the univariate Gaussian case: let $z \sim p(x|x) = \mathcal{N}(\mu, \sigma^2)$. In this case, a valid re-parameterization is $z = \mu + \sigma\epsilon$, where $\epsilon$ is an auxiliary noise variable $\epsilon \sim \mathcal{N}(0,1)$.

### Experiments

> We trained generative models of images from the MNIST and Frey Face datasets and compared learning algorithms in terms of the variational lower bound, and the estimated marginal likelihood.

![Screenshot 2024-05-18 at 11.58.38 AM.png](../../images/Screenshot_2024-05-18_at_11.58.38_AM.png)

### Conclusion

> We have introduce a novel estimator of the variational lower bound, Stochastic Gradient VB (SGVB), for efficient approximate inference with continuous latent variables.

> The theoretical advantages are reflected in experimental results.
