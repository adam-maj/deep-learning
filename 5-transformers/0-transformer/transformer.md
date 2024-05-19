# Transformer

📜 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

> We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

Building on the state of encoder-decoder architectures, the Transformer removes all the complexities and uses attention as the only source of computation.

> On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

This is an insane result showing the huge efficiency jump due to parallelization that the transformer offers - better performance in a fraction of the time.

> Recurrent models typically factor computation along the symbol positions of the input and output sequences. […] The fundamental constraint of sequential computation, however, remains.

The fundamental computational constraints of recurrent models makes them inefficient, despite optimization attempts.

> In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

Motivated by the inefficiency of recurrence, this paper completely gets rid of recurrence and uses only attention to form representations.

> The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

> In the Transformer [the number of operations required to relate signals from two arbitrary input or output positions] is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention

In the Transformer, attention in a single head takes an average of the relevant information at all positions to compute the context to update individual word embedding vectors with, which can have the effect of reduce resolution. To counter-act this, multi-headed attention creates another way for the transformer to focus on multiple different important types of information for each word.

> Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

Individual words in a sequence can soak up context from each other to enrich their own meanings.

### Model Architecture

> The Transformer follows this overall architecture [of an encoder-decoder model] using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

![Screenshot 2024-05-15 at 11.49.10 PM.png](../../images/Screenshot_2024-05-15_at_11.49.10_PM.png)

**1. Encoder and Decoder Stacks**

> The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

Each of the $N$ layers contains a multi-headed attention block for the words in the input sequence to self-attend to each other and soak up meanings from each other, as well as a feed forward network to use and interpret those meanings.

Additionally, each sub-layer uses layer normalization and residuals for optimization purposes.

> The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

Masked multi-head attention is used in the decoder to ensure that output words can’t attend to words that follow them

**2. Attention**

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

**2.1. Scaled Dot-Product Attention**

![Screenshot 2024-05-15 at 11.54.43 PM.png](../../images/Screenshot_2024-05-15_at_11.54.43_PM.png)

```math
\textrm{Attention}(Q,K,V) = \textrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
```

Both dot-product and additive attention were viable choices. However:

> Dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

> We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has
> extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

The scaling factor is just to keep the outputs of the softmax function in the regime where it’s gradient doesn’t vanish, regardless of the dimensions size (by minimizing the differences between values).

**2.2. Multi-Head Attention**

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

The primary affect of multi-headed attention is to enable the model to soak up context from multiple distinct potential information sources for each word - something that couldn’t happen as easily with just a single head due to the fact that individual heads take weighted averages of the

```math
\textrm{MultiHead}(Q,K,V) = \textrm{Concat}(\textrm{head}_1,...,\textrm{head}_h)W^O \\
\textrm{where head}_i = \textrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

Each multi-headed attention block concatenates the results of each of its head into the final vector that’s passed to the feed-forward layer.

This means that the feed-forward layer is actually not just receiving a single opinion about the words enriched context, but actually a completely different perspective about the words meaning for each head.

In this case, that means that the feed-forward layer is actually receiving 8 different perspectives on the enriched meaning of each token.

> Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

**2.3. Applications of Attention in our Model**

> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

```math
\textrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
```

> While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.

Because the same weight matrix is applied to each enhanced token outputted from the multi-headed attention layer, this can be thought of as a convolution being applied to each position.

**4. Embeddings and Softmax**

> We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

**5. Positional Encodings**

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

This is what enables Transformers to still maintain information about word positions without explicit recurrence.

> In this work, we use sine and cosine functions of different frequencies:

```math
PE_{(pos,  2i)} = \sin(pos/1000^{2i/d_{\textrm{model}}}) \\
PE_{(pos,  2i+1)} = \cos(pos/1000^{2i/d_{\textrm{model}}})
```

### Why Self-Attention

> Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. The third is the path length between long-range dependencies in the network.

> Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.

Transformers make it far easier for the network to learn long-range dependencies between words since there’s no recurrence to make path lengths across time long.

> To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position.

This is the motivation behind limited context windows.

> As side benefit, self-attention could yield more interpretable models. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

Multi-headed attention appears to model highly interpretable behaviors, something uncharacteristic of most models.

### Training

> We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.

> Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens.

> We trained our models on one machine with 8 NVIDIA P100 GPUs. […] Each training step took about 0.4 seconds.

> We used the Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{−9}$.

> We apply dropout to the output of each sub-layer before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.

> During training, we employed label smoothing of value $\epsilon_{l_s} = 0.1$.

### Results

> Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

> To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes.

> Despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar.

### Conclusion

> In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

> For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.
