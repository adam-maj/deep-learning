# ReLU

📜 [Deep Sparse Rectified Neural Networks](https://www.researchgate.net/publication/215616967_Deep_Sparse_Rectifier_Neural_Networks)

> Many differences exist between the neural network models used by machine learning researchers and those used by computational neuroscientists.

This paper highlights the importance of areas where both neuroscience and machine learning researches are both interested, which point toward “computationally motivated principles of operation in the brain that can also enhance research in artificial intelligence”

ReLU apparently bridges two of the gaps between neuroscience and practical ML.

The backdrop of this paper is a world where an unsupervised pre-training step for all supervised learning tasks helped with weight initialization in regularization and these networks (like deep belief networks) performed significantly better on supervised learning tasks than networks without this step.

ReLU is an attempt to close this gap in performance.

### Background

**1. Neuroscience Observations**

> Studies on brain energy expense suggest that neurons encode information in a sparse and distributed way.

Only around 1-4% of neurons are active at a time. This is a trade-off between “richness of representation” and low energy expenditure for action potentials.

The sigmoid activation does not enable this, as it has a “steady-state regime around 1/2,” (small inputs get squashed to around 0.5) so all neurons fire at their saturation, which is biologically implausible and hurts optimization.

Meanwhile, a common biological neuron activation function is the leaky integrate-and-fire which has a zero threshold.

![Screenshot 2024-05-13 at 10.59.21 AM.png](../../images/Screenshot_2024-05-13_at_10.59.21_AM.png)

**2. Sparsity**

> We show here that using a rectifying non-linearity gives rise to real zeros of activations and thus truly sparse representations.

This has the following benefits:

(1) Information disentangling

> One of the claimed objectives of deep learning algorithms is to disentangle the factors explaining the variations in the data.

Without zeroing weights, any change in input modifies all parts of the representation. If the representation is instead sparse and robust to small input changes, the set of non-zero features is mostly conserved for small changes.

(2) Efficient variable-size representation

> Varying the number of active neurons allows a model to control the effective dimensionality of the representation for a given input and the required precision.

Different inputs may contain a different about of information and should be represented by a variable size vector. Sparse coding allows the network to adapt to this need.

(3) Linear separability

> Sparse representations are also more likely to be linearly separable

Which is good because linear separation is computationally simpler of a decision function to learn

(4) Distributed but sparse

> Dense distributed representations are the richest representations, being potentially exponentially more efficient than purely local ones. Sparse representations’ efficiency is still exponentially greater, with the power of the exponent being the number of non-zero features.

Less activations to compute in feed-forward means less computation. But more importantly, more zeroed values means tons of values don’t need to be computed in back-propagation which is efficient.

> Nevertheless, forcing too much sparsity may hurt predictive performance for an equal number of neurons, because it reduces the effective capacity of the model.

### Deep Rectifier Networks

**1. Advantages of Rectifier Neurons**

The rectifier activation function $\textrm{rectifier}(x) = \textrm{max}(0,x)$ allows a network to easily obtain sparse representations.

For example, on initialization ~50% of the hidden units output values will be zeroes.

> Apart from being more biologically plausible, sparsity also leads to mathematical advantages.

> For a given input, only a subset of neurons are active. Computation is. linear on this subset. Once the subset of neurons is selected, the output is a linear function of the input.

> We can see the model as an _exponential number of linear models that share parameters_.

> Because of this linearity, gradients flow well on the active paths of neurons (there is no gradient vanishing effect due to the activation non-linearities of sigmoid or tanh units).

> Computations are also cheaper: there is no need for computing the exponential function in activations, and sparsity can be exploited.

ie. the derivative of ReLU is computationally cheaper than tanh and sigmoid when you do have to calculate it, and most of the outputs are zeroed anyway so you don’t even have to calculate them.

**2. Potential Problems**

> One may hypothesize that the hard saturation at 0 may hurt optimization by blocking gradient back-propagation

They tried soft-plus as a way to test this. It’s not an issue - hard zeros actually help supervised training.

> We hypothesize that the hard non-linearities do not hurt _so long as the gradient can propagate along some paths_.

> With the _credit and blame assigned to these ON units_ rather than distributed more evenly, we hypothesize that optimization is easier.

> Another problem could arise due to the unbounded behavior of activations; one may thus want to use a regularizer to prevent potential numerical problems.

So they use the $L_1$ penalty on the activation values which also promotes additional sparsity.

### Conclusion

> Sparsity and neurons operating mostly in a linear regime can be brought together in more biologically plausible deep neural networks

> Rectifier units help to bridge the gap between unsupervised pre-training and no pre-training, which suggests that they may help in finding better minima during training.

> Furthermore, rectifier activation functions have shown to be remarkably adapted to sentiment analysis, a text-based task with a very large degree of data sparsity.
