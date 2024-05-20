# GELU

📜 [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415)

> Despite having less of a statistical motivation, the ReLU remains a competitive engineering solution which often enables faster and better convergence than sigmoids.

> Nonlinearities and dropout thus determine a neuron’s output together, yet the two innovations have remained distinct.

> [GELU] relates to stochastic regularizers in that it is the expectation of a modification to Adaptive Dropout.

### GELU Formulation

> We motivate our activation function by combining properties from dropout, zone-out, and ReLUs.

The paper introduces a stochastic regularizer that multiplies a neuron input $x$ by $m \sim \text{Bernoulli}(\Phi(x))$ where $\Phi(x)$ is the CDF of the Gaussian distribution (standard normal distribution).

> We choose this distribution since neuron inputs tend to follow a normal distribution, especially with Batch Normalization.

> In this setting, inputs have a higher probability of being “dropped” as x decreases, so the transformation applied to x is stochastic yet depends upon the input.

Here, we have something similar to adaptive dropout, where the probability of dropout is not constant, but instead is dependent on the input value (more “important” inputs, as determined their relative scale after normalization are less likely to be dropped out).

In order to motivate our non-linearity, we create a deterministic version of this function $\textrm{GELU}(x) = x \Phi(x)$.

With this function, $x$ gets closer to the identity as the CDF gets higher, and $x$ gets closer to being zeroed as the CDF gets lower. Asymptotically, this distribution behaves the same as ReLU.

> Loosely, this expression states that we scale x by how much greater it is than other inputs.

Basically, $x$ gets prioritized to be mapped to the identity function if it is originally much larger than the other inputs to normalization, and thus is normalized to a higher value.

### Discussion

> For example, as σ → 0 and if µ = 0, the GELU becomes a ReLU.

> More, the ReLU and GELU are equal asymptotically

> In fact, the GELU can be viewed as a way to smooth a ReLU.

> This non-convex, non-monotonic function is not linear in the positive domain and exhibits curvature at all points. Meanwhile ReLUs and ELUs, which are convex and monotonic activations, are linear in the positive domain and thereby can lack curvature. As such, increased curvature and non-monotonicity may allow GELUs to more easily
> approximate complicated functions than can ReLUs or ELUs.

> We can see that the ReLU gates the input depending upon its sign, while the GELU weights its input depending upon how much greater it is than other inputs.

> In addition and significantly, the GELU has a probabilistic interpretation given that it is the expectation of a stochastic regularizer.

### Conclusion

> For the numerous datasets evaluated in this paper, the GELU exceeded the accuracy of the ELU and ReLU consistently, making it a viable alternative to previous nonlinearities.
