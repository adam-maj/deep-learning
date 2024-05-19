# Vision Transformer

📜 [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

> When trained on mid-sized datasets such as ImageNet without strong regularization, these [transformer] models yield modest accuracies of a few percentage points below ResNets of comparable size

> Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.

> However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias.

Another huge statement again showing the scale is all you need. The constraint is really how much is a model architecture conducive to scale (which is constrained by optimization & regularization with large number of parameters, compute, and time to convergence).

The inductive bias of CNNs actually doesn’t get you as far as training transformers on more data.

> Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus, to apply Transformers in the context of image processing, several approximations have been tried in the past.

### Method

**1. Vision Transformer**

> In model design we follow the original Transformer as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box

Don’t reinvent the wheel. Do as little work as possible to verify just your core thesis and not build everything from scratch.

The image is divided into patches of dimension $(P, P)$ where there are $N = HW/P^2$ total patches, which is also the number of inputs to the transformer.

The transformer uses constant latent vector size $D$ for every layer, so we create a projection from the $P^2$ length flattened patches to the size $D$ embedding vector for each patch. This projection is trained with the network.

Positional encodings are also included with each token, although they include no information about the patches 2D position.

**2. Inductive Bias**

> We note that Vision Transformer has much less image-specific inductive bias than CNNs.

Transformers has almost no inductive bias compared with what the CNNs have, but they still perform better. Needing inductive bias is a sign that you haven’t scaled far enough.

### Conclusion

> Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step. Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP.

> This simple, yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets. Thus, Vision Transformer matches or exceeds the state of the art on many image classification datasets, whilst being relatively cheap to pre-train.

> Finally, further scaling of ViT would likely lead to improved performance.
