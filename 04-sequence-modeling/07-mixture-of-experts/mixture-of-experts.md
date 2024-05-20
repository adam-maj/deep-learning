# Mixture of Experts

📜 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538)

> The capacity of a neural network to absorb information is limited by its number of parameters.

> Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation.

> In this work, we […] finally realize the promise of conditional
> computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency.

> We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example.

> Various forms of conditional computation have been proposed as a way to increase model capacity without a proportional increase in computational costs.

> In these schemes, large parts of a network are active or inactive on a per-example basis.

> Our approach to conditional computation is to introduce a new type of general purpose neural network component: a Sparsely-Gated Mixture-of-Experts Layer (MoE). The MoE consists of a number of experts, each a simple feed-forward neural network, and a trainable gating network which selects a sparse combination of the experts to process each input.

### The Structure of the Mixture-of-Experts Layer

> The Mixture-of-Experts (MoE) layer consists of a set of $n$ “expert networks” $E_1, …, E_n$ and a “gating network” $G$ whos output is a sparse $n$-dimensional vector.

![Screenshot 2024-05-17 at 3.30.44 PM.png](../../images/Screenshot_2024-05-17_at_3.30.44_PM.png)

> Let us denote by $G(x)$ and $E_i(x)$ the output of the gating network and the output of the $i$-th expert network for a given input $x$. The output $y$ of the MoE module can be written as follows:

```math
y = \sum_{i=1}^n G(x)_i E_i(x)
```

> We save computation based on the sparsity of the output of $G(x)$. Wherever $G(x)_i$ = 0, we need not compute $E_i(x)$. In our experiments, we have up to thousands of experts, but only need to evaluate a handful of them for every example.

Because of the gating mechanism, only a fraction of experts are actually used in each training run, meaning that while the total network may have $N \times P$ total parameters, only $n \times P$ need to be optimized on every training run where $n$ is the number of active experts at once, meaning setting $n \ll N$ saves significant compute, while maintaining model size.

Hierarchical MoE also exists, further improving compute.

**1. Gating Network**

A simple approach for gating would be to use a trainable weight matrix $W_g$ for gating with the softmax function

```math
G_\sigma(x) = Softmax(x \cdot W_g)
```

> We add two components to the Softmax gating network: sparsity and noise. Before taking the softmax function, we add tunable Gaussian noise, then keep only the top k values, setting the rest to $-\infty$.

```math
G(x) = Softmax(KeepTopK(H(x), k)) \\

H(x)_i = (x \cdot W_g)_i + Gaussian() \cdot Softplus((x \cdot W_{noise})_i)
```

### Addressing Performance Challenges

**1. The Shrinking Batch Problem**

> If the gating network chooses $k$ out of $n$ experts for each example, then for a batch of $b$ examples, each expert receives a much smaller batch of approximately $\frac{kb}{n} \ll b$ examples. This causes a naive MoE implementation to become very inefficient as the number of experts increases.

> The solution to this shrinking batch problem is to make the original batch size as large as possible. However, batch size tends to be limited by the memory necessary to store activations between the forwards and backwards passes.

This mixture-of-experts implementation uses a form of data-parallelism where each expert lives on every device, meaning that it gets trained on the batches of data sent to each device, synchronously.

This adds a $d$ term, making the total amount of batch data training each expert $\frac{kbd}{n}$.

**2. Network Bandwidth**

> Another major performance concern in distributed computing is network bandwidth.

> To maintain computational efficiency, the ratio of an expert’s computation to the size of its input and output must exceed the ratio of computational to network capacity of the computing device.

### Balancing Expert Utilization

> We have observed that the gating network tends to converge to a state where it always produces large weights for the same few experts.

If left unattended, expert networks will naturally prioritize certain experts, and will keep training and improving the same ones.

To ensure equal importance of all experts across the batch, we introduce an additional importance term to the loss function.

```math
Importance(X) = \sum_{x \in X} G(x) \\

L_{importance}(X) = w_{importance} \cdot CV(Importance(X))^2
```

This function calculates the total importance of each expert across a batch by summing the gates for that expert for each training example, taking the square of the coefficient of variation, and multiplying it by a scaling factor $w_{importance}$.

### Experiments

![Screenshot 2024-05-17 at 3.59.23 PM.png](../../images/Screenshot_2024-05-17_at_3.59.23_PM.png)

![Screenshot 2024-05-17 at 3.59.51 PM.png](../../images/Screenshot_2024-05-17_at_3.59.51_PM.png)

### Conclusion

> This work is the first to demonstrate major wins from conditional computation in deep networks.
