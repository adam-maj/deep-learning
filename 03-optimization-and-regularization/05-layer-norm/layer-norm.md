# Layer Normalization

📜 [Layer Normalization](https://arxiv.org/pdf/1607.06450)

> The effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks.

> Unlike batch normalization, layer normalization performs exactly the same computation at training and test times.

> It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step.

> It is possible to speed-up the learning by computing gradients for different subsets of the training cases on different machines or splitting the neural network itself over many machines.

> An orthogonal approach is to modify the computations performed in
> the forward pass of the neural net to make learning easier.

Work hard vs. work smart. You can throw more compute at your current architecture, but you can also improve your architecture to make training more efficient. Architectural efficiency increases the leverage on your compute.

> The summed inputs to the recurrent neurons in a recurrent neural network (RNN) often vary with the length of the sequence so applying batch normalization to RNNs appears to require different statistics for different time-steps.

For RNNs, you can’t normalize across the entire input space, as inputs at each time-step are completely different.

Batch normalization requires normalizing across an input space where normalization is actually statistically coherent. Normalizing across all inputs in RNNs does not make sense.

> Furthermore, batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the mini-batches have to be small.

Batch normalization doesn’t work in _online learning_ cases where data isn’t processed in batches, and also in distributed models where mini-batches are kept small due to limited memory, which affects the statistical accuracy of normalizing across a mini-batch as an approximation for the distribution of the whole dataset.

### Layer Normalization

> We now consider the layer normalization method which is designed to overcome the drawbacks of batch normalization.

> The “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer.

As this also acts as an equivalent way to make sure that the input distributions of the next layer are more consistent, which is the real end goal.

**1. Layer normalized recurrent neural networks**

Batch normalization in recurrent networks cause problems since sequences of different sentence lengths are often used for different training cases.

> Layer normalization does not have such problem because its normalization terms depend only on the summed inputs to a layer at the current time-step.

Layer normalization doesn’t face these issues and makes normalization compatible with recurrence.

> In a standard RNN, there is a tendency for the average magnitude of the summed inputs to the recurrent units to either grow or shrink at every time-step, leading to exploding or vanishing gradients.

> In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed inputs to a layer, which results in much more stable hidden-to-hidden dynamics.

### Analysis

**1. Invariance under weights and data transformations**

> [All normalizations] can be summarized as normalizing the summed inputs $a_i$ to a neuron through the two scalars $\mu$ and $\sigma$. They also learn an adaptive bias $b$ and gain $g$ for each neuron after the normalization.

The final model of a neurons activation is $h_i = f(\frac{g_i}{\sigma_i}(a_i - \mu_i) + b_i)$ where $f$ is the activation function.

In layer normalization and batch normalization, $\mu$ and $\sigma$ are computed as mean and variance. In weight normalization, $\mu = 0$ and $\sigma$ is the $L_2$ norm of the weights.

**2. Weight re-scaling and re-centering**

> First, observe that under batch normalization and weight normalization, any re-scaling to the incoming weights $w_i$ of a single neuron has no effect on the normalized summed inputs to a neuron.

Scaling incoming weights has no effect due to the fact that individual neurons are normalized across examples in batch normalization, so changes in weights have an equivalent effect on all inputs to normalization.

> Layer normalization, on the other hand, is not invariant to the individual scaling of the single weight vectors.

This is because as weights of one neuron change, the activations get relatively larger, causing them to be scaled differently.

> Instead, layer normalization is invariant to scaling of the entire weight matrix and invariant to a shift to all of the incoming weights in the weight matrix.

Absolute weight changes don’t matter, but relative weight changes do.

> Notice that if normalization is only applied to the input before the weights, the model will not be invariant to re-scaling and re-centering of the weights.

**3. Data re-scaling and re-centering**

> All the normalization methods are invariant to re-scaling the dataset.

**3. Geometry of parameter space during learning**

> We show that the normalization scalar $\sigma$ can implicitly reduce learning rate and makes learning more stable.

> The normalization methods, therefore, have an implicit “early stopping” effect on the weight vectors and help to stabilize learning towards convergence.

> Learning the magnitude of incoming weights in the normalized model is therefore, more robust to the scaling of the input and its parameters than in the standard model.

### Experimental Results

> Unless otherwise noted, the default initialization of layer normalization is to set the adaptive gains to 1 and the biases to 0 in the experiments.

![Screenshot 2024-05-13 at 1.27.53 PM.png](../../images/Screenshot_2024-05-13_at_1.27.53_PM.png)

Here we see LN more efficient than BN and base model in converging.

They proceed to cove many language experiments where Layer Normalization is empirically the best approach.

> We have also experimented with convolutional neural networks. In our preliminary experiments, we observed that layer normalization offers a speedup over the baseline model without normalization, but batch normalization outperforms the other methods.

For now, it appears that Layer Normalization is more effective than Batch Normalization primarily in sequence-based language-modeling tasks.

### Conclusion

> We provided a theoretical analysis that compared the invariance properties of layer normalization with batch normalization and weight normalization.

> We showed that layer normalization is invariant to per training-case feature shifting and scaling.

> Empirically, we showed that recurrent neural networks benefit the most from the proposed method especially for long sequences and small mini-batches.
