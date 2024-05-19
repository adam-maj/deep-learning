# Diffusion

📜 [Deep Unsupervised Learning Using Non-equilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585)

> A central problem in machine learning involves modeling complex data-sets using highly flexible families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally tractable.

This captures the challenge that all the different probabilistic/generative models have to solve.

> The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data.

> Historically, probabilistic models suffer from a tradeoff between two conflicting objectives: tractability and flexibility.

**1. Diffusion Probabilistic Models**

> We present a novel way to define probabilistic models that allows:

1. extreme flexibility in model structure
2. exact sampling
3. easy multiplication with other distributions, e.g. in order to compute a posterior, and
4. the model log likelihood, and the probability of individual states, to be cheaply evaluated
   >

> Our method uses a Markov chain to gradually convert one distribution into another, an idea used in non-equilibrium statistical physics.

The diffusion process slowly converts distributions from more noisy to more structured forms.

> Since a diffusion process exists for any smooth target distribution, this method can capture data distributions of arbitrary form.

### Algorithm

> Our goal is to define a forward (or inference) diffusion process which converts any complex data distribution into a simple, tractable, distribution, and then learn a finite-time reversal of this diffusion process which defines our generative model distribution.

**1. Forward Trajectory**

> The data distribution is gradually converted into a well behaved distribution $\pi (y)$ by repeated application of a Markov diffusion kernel $T_\pi(y|y’; \beta)$ for $\pi(y)$ where $\beta$ is the diffusion rate.

```math
\pi(y) = \int dy' T_\pi(y|y';\beta)\pi(y') \\

q(x^{(t)}|x^{(t-1)}) = T_\pi(x^{(t)}|x^{(t-1)}; \beta_t)
```

> The forward trajectory corresponding to starting at the data distribution and performing $T$ steps of diffusion is thus

```math
q(x^{(0...T)}) = q(x^{(0)}) \prod_{t=1}^T q(x^{(t)}|x^{(t-1)})
```

**2. Reverse Trajectory**

> The generative distribution will be trained to describe the same trajectory, but in reverse.

```math
p(x^{(T)}) = \pi(x^{(T)}) \\
p(x^{(0...T)} = p(x^{(T)}) \prod_{t=1}^T p(x^{(t-1)}|x^{(t)})
```

**3. Model Probability**

> The probability the generative model assigns to the data is

```math
p(x^{(0)}) = \int dx^{(1...T)} p(x^{(0...T)})
```

In order to actually be able to calculate this integral

> We can instead evaluate the relative probability of the forward and reverse trajectories, averaged over forward trajectories.

```math
p(x^{(0)}) = \int dx^{(1...T)}q(x^{(1...T)}|x^{(0)}) \cdot p(x^{(T)}) \prod_{t=1}^T \frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}
```

**4. Training**

Training amounts to maximizing the model log likelihood

```math
L = \int dx^{(0)} q(x^{(0)}) \log p(x^{(0)})
```

Here, we maximize the likelihood that the model $p$ generates the state $x^{(0)}$ from some noisy state, conditioned by the weight of the actual sample in the dataset $q(x^{(0)})$.

> The derivation of this bound parallels the derivation of the log likelihood bound in variational Bayesian methods.

### Experiments

> We train diffusion probabilistic models on a variety of continuous datasets, and a binary dataset. We then demonstrate sampling from the trained model and in-painting of missing data, and compare model performance against.

![Screenshot 2024-05-18 at 3.10.57 PM.png](../../images/Screenshot_2024-05-18_at_3.10.57_PM.png)

### Conclusion

> We have introduced a novel algorithm for modeling probability distributions that enables exact sampling and evaluation of probabilities and demonstrated its effectiveness on a variety of toy and real datasets, including challenging natural image datasets.

> The result is an algorithm that can learn a fit to any data distribution, but which remains tractable to train, exactly sample from, and evaluate, and under which it is straightforward to manipulate conditional and posterior distributions.
