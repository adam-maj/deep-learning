# RNN

📜 [A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)

> A major problem in connectionist theory is to develop learning algorithms that can tap the full computational power of neural networks.

How do we use the full potential of neural networks? Designing architectures that enables this for different problems is challenging.

> Attention has recently turned to developing algorithms for networks with recurrent connections, which have important capabilities not found in feedforward networks, including attractor dynamics and the ability to store information for later use.

> The approach we propose here enjoys the generality of the backpropagation-through-time approach while not suffering from its growing memory requirement in arbitrarily long training sequences

### The Learning Algorithm and Variations

**1. The Basic Algorithm**

We define a network with $n$ output units, with $m$ input lines. We create the $(m + n)$-tuple $z(t)$, the concatenation of the set of input signals at time $t$, $x(t)$ and the set of output signals at time $t$, $y(t)$. Let the indices $k \in I$ represent the input units and $k \in U$ represent the output units.

```math
z_k(t) =
\begin{cases}
  x_k(t) & \textrm{if } k \in I \\
  y_k(t) & \textrm{if } k \in U
\end{cases}
```

Let the activation of the $k$th unit at time $t$ for $k \in U$ be defined by

```math
y_k(t + 1) = f_k(s_k(t))
```

With $f_k$ representing the neurons activation function and $s_k$ representing the sum of the neurons weights as defined by

```math
s_k(t) = \sum_{i \in U \cup I} w_{ki} z_i(t)
```

And the notation $y_k(t+1)$ denoting that the neurons output at time-step $t$ will also be it’s contributing input to the network at time step $t + 1$.

Given $T(t)$ as the subset of neurons $k \in U$with a specific target output value $d_k(t)$ for neuron $k$ at time $t$. We can then define the error function:

```math
e_k(t) =
\begin{cases}
  d_k(t) - y_k(t) & \textrm{if } k \in T(t) \\
  0 & \textrm{otherwise}
\end{cases}
```

> Note that this formulation allows for the possibility that target values are specified for different units at different times. The set of units considered to be visible can thus be time-varying,

Now the error function to compute the total error of the network at time $t$

```math
J(t) = 1/2 \sum_{k \in U}[e_k(t)]^2
```

And the overall weight change for a weight $w_{ij}$ in the network

```math
\Delta w_{ij}(t) = -\alpha\frac{\partial J(t)}{\partial w_{ij}}
```

```math
\frac{\partial{J(t)}}{\partial{w_{ij}}} = \sum_{k \in U} e_k(t) \frac{\partial{y_k(t)}}{\partial{w_{ij}}}
```

```math
\frac{\partial{y_k(t+1)}}{\partial{w_{ij}}} = f'_k(s_k(t))[\sum_{i \in U} w_{ki} \frac{\partial{y_i(t)}}{\partial{w_{ij}}} + \delta_{ik}z_j(t)]
```

These all obtained just through simple back-propagation on the original setup of our network using the chain-rule. The last $\frac{\partial{y_k(t+1)}}{\partial{w_{ij}}}$ representing the partial of the output neurons of the _current_ time-step with respect to the weights and outputs of the _previous_ time-step, thus enabling back-propagation recursively through time.

Then for convenience we define $p_{ij}^k$ where $k \in U$, $i \in U$, $j \in U \cup I$, $k$ denoting the output neuron who’s gradient to back-propagate to the _previous_ time-step, and $i$, $j$ specifying the weight between an input and output neuron in the previous time-step.

```math
p_{ij}^k(t+1) = f'_k(s_k(t))[\sum_{l \in U} w_{kl} p_{ij}^l(t) + \delta_{ik} z_j(t)]
```

Thus, the gradient recursively back-propagates through previous time-steps via the recursive use of $p_{ij}^l$ in it’s own definition, hitting a base case at $t = 0$ where we define the the value to be 0 since the output neuron values at this point are un-tethered to any previous time-step.

```math
p_{ij}^k(t_0) = 0
```

Then, we have the final weight update algorithm at a given time-step $t$

```math
\Delta w_{ij}(t) = \alpha \sum_{k \in U} e_k(t) p_{ij}^k (t)
```

**2. Real-Time Recurrent Learning**

> In order to allow real-time training of behaviors of indefinite duration, however, it is useful to relax this assumption and actually make the weight changes while the network is running.

In practice, we update weights at every time-step, rather than accumulating gradients and then changing them at epochs.

> While the resulting algorithm is no longer guaranteed to follow the gradient of total error, the practical differences are often slight, with the two versions becoming more nearly identical as the learning rate is made smaller.

**3. Teacher-Forced Real-Time Recurrent Learning**

> An interesting technique that is frequently used in dynamical supervised learning tasks is to replace the actual output $y_k(t)$ of a unit by the teacher signal $d_k(t)$ in subsequent computation of the behavior of the network, whenever such a value exists. We call this technique _teacher forcing_.

### Discussion

> Our primary goal here has been to derive a learning algorithm to train completely recurrent, continually updated networks to learn temporal tasks.

> Our emphasis has been on using uniform starting configurations that contain no a priori information about the temporal nature of the task.

> The solutions found by the algorithm are often dauntingly obscure, particularly for complex tasks involving internal state. This observation is already familiar in work with feedforward networks. This obscurity has often limited our ability to analyze the solutions in sufficient detail.

Interpretability has been near impossible, even from the beginning.
