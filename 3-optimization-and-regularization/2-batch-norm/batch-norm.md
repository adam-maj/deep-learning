# Batch Normalization

📜 [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change.

> This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating non-linearities.

Later layers depend on previous layers for their inputs, meaning the input distribution changes very quickly for later layers as earlier layers learn - potentially rendering much of their earlier learning useless.

This means models require lower learning rates so small changes can be made over time.

> We refer to this phenomenon as **internal covariate shift**, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch.

Normalization is part of the actual model rather than just a weight decay penalty on the cost function.

> **Batch Normalization** allows us to use much higher learning rates and
> be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout.

Deep learning using SGD with mini-batches has become popular, as approximating gradient steps across batches is more efficient than individual examples due to parallelization and better estimation of correct gradients for the whole dataset (which increases with batch size)

However, the internal covariate shift problem strikes here, forcing us to use low learning rates.

Additionally, the distribution of nonlinearities can easily get stuck in the high saturation regime (weight decay helps with this somewhat).

> If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

> Eliminating [internal covariate shift] offers a promise of faster training.

This is because you can use higher learning rates since there’s less overshooting and incorrect optimization due to shifting input distributions.

> **Batch normalization** dramatically accelerates the training of deep neural nets. It accomplishes this via a normalization step that fixes the means and variances of layer inputs.

> Batch normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of parameters or of their initial values. This allow us to use much higher learning rates without the risk of divergence.

Scaling to the same input distribution means that parameter scales don’t vary too much, meaning gradients don’t get outsized affects to parameter scales as much as they do to the relative importance of features.

> Furthermore, batch normalization regularizes the model and reduces the need for Dropout

> Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes.

### Towards Reducing Internal Covariate Shift

> We would like to ensure that, for any parameter values, the network _always_ produces activations with the desired distribution.

We could propose a normalization $\hat{x} = \textrm{Norm}(x, X)$ that normalizes not just based on the current input but normalizes it relative to the entire dataset, meaning gradients would need to be computed against the entire dataset.

> We want to preserve the information in the network, by normalizing the activations in a training example relative to the statistics of the entire training data.

### Normalization via Mini-Batch Statistics

Since normalizing across each layer’s inputs across the entire dataset is costly, we make two simplifications.

> The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and variance of 1.

![Screenshot 2024-05-13 at 12.11.52 PM.png](../../images/Screenshot_2024-05-13_at_12.11.52_PM.png)

Here, we normalize the inputs to each layer at the level of each neuron, across training examples from the entire dataset.

> Note that simply normalizing each input of a layer may change what the layer can represent.

This is because squashing the input values to the activation function to a limited range can limit expression of activation functions to only specific features - for example in sigmoid, squashing would “constrain activations to the linear regime of the nonlinearity.”

> To address this, we make sure that _the transformation in the network can represent the identity transform._

![Screenshot 2024-05-13 at 12.16.01 PM.png](../../images/Screenshot_2024-05-13_at_12.16.01_PM.png)

So the model can learn some transform back on the batch normalized inputs before passing through activation that can eventually rescale them back to their original values if necessary, by setting $\gamma^{(k)} = \sqrt{\textrm{Var}[x^{(k)}]}$ and $\beta^{(k)} = \textrm{E}[x^{(k)}]$ to reverse the effect of the normalization.

Additionally, doing this entire normalization across the whole dataset would be expensive.

> Therefore, we make the second simplification: since we use mini-batches in stochastic gradient training, _each mini-batch produces estimates of the mean and variance_ of each activation.

In contrast to previous normalization work, this approach does _not_ use joint covariance matrices, which is convenient since the mini-batch size is usually smaller than the number of neurons in a layer, which would cause issues in the calculation of the inverse.

> $\textrm{BN}_{\gamma, \beta}(x)$ depends on the training example and the other examples in the mini-batch.

> This ensures that as the model is training, layers can continue learning on input distributions that exhibit less internal covariate shift, thus accelerating the training.

**1. Training and Inference with Batch Normalized Networks**

> The normalization of activations that depends on the mini-batch allows efficient training, but is neither necessary nor desirable during inference.

Once training is complete, we just use normalization across the entire population, rather than mini-batch statistics.

> Since the means and variances are fixed during inference, the normalization is simply a linear transform applied to each activation.

**3. Batch Normalization Enables Higher Learning Rates**

> By normalizing activations throughout the network, it prevents small changes to the parameters from amplifying into larger and suboptimal changes in activations and gradients.

> Batch normalization also makes training more resilient to the parameter scale. Normally, learning rates may increase the scale of layer parameters, which then amplify the gradient back-propagation and lead to the model explosion.

Batch normalization does not do this, as scaling parameters does not affect it’s outputs $\textrm{BN}(Wu) = \textrm{BN}((aW)u)$.

**4. Batch Normalization Regularizes the Model**

> When training with Batch Normalization, a training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing deterministic values for a given training example

The outputs for a single input in training are non-deterministic on the model weights because the other inputs in their batch affect their activations - this helps with regularization.

> Whereas Dropout is typically used to reduce overfitting, in a batch-normalized network, we found that it can be either removed or reduced in strength.

### Experiments

**1. Activations Over Time**

Batch normalization networks for MNIST converged far faster, and have higher test accuracy overall.

![Screenshot 2024-05-13 at 12.41.37 PM.png](../../images/Screenshot_2024-05-13_at_12.41.37_PM.png)

**2. Accelerating BN networks**

Just adding Batch Normalization to a network does not take full advantage of what it has to offer. To maximize improvements and efficiency, the following steps were taken:

(1) _Increase learning rate with no ill side effects_

(2) _Remove dropout_ - BN speeds up training without increasing overfitting

(3) _Reduce the $L_{2}$ weight regularization\_ - since weight decay isn’t needed as much since activations are already scaled to be low

(4) _Accelerate the learning rate decay_

(5) R*emove local response normalization* - in CNNs, can reduce local response normalization (lateral inhibition) as normalization already accomplishes this.

(6) _Shuffle training examples more thoroughly_ - this is consistent with the model of BN as a regularizer.

(7) _Reduce the photometric distortions_ - since BN allows the networks to train faster and observe each training example fewer times, we let the model focus more on real images and don’t distort them as much.

### Conclusion

> We have presented a novel mechanism for dramatically accelerating the training of deep networks. It is based on the premise that covariate shift, which is known to complicate the training of machine learning systems, also applies to sub-networks and layers, and removing it from internal activations of the network may aid in training.

> Our proposed method draws its power from normalizing activations, and from incorporating this normalization in the network architecture itself. This ensures that the normalization is appropriately handled by any optimization method that is being used to train the network.

Important that the normalization is part of the network so back-propagation has access to optimizing it’s parameters.

> Batch Normalization adds only two extra parameters per activation, and in doing so preserves the representation ability of the network.

> Merely adding Batch Normalization to a state-of-the art image classification model yields a substantial speedup in training.

> By further increasing the learning rates, removing Dropout, and applying other modifications afforded by Batch Normalization, we reach the previous state of the art with only a small fraction of training steps - and then beat the state of the art in single-network image classification.

> Furthermore, by combining multiple models trained with Batch Normalization, we perform better than the best known system on ImageNet, by a significant margin.
