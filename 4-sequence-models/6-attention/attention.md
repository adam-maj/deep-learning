# Attention

📜 [Neural Machine Translation By Jointly Learning To Align And Translate](https://arxiv.org/pdf/1409.0473)

> The models proposed recently for neural machine translation often belong to a family of encoder–decoders and encode a source sentence into a fixed-length vector from which a decoder generates a translation.

> In this paper, we conjecture that the use of a fixed-length vector is a
> bottleneck in improving the performance of this basic encoder–decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly.

We introduce the attention mechanism to enable a word to search in previous context for information related to it.

> Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition.

The attention mechanism appears to be discovering intuitive relationships between words.

> A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector.

This limits the length of sentences and context that the encoder can actually represent effectively in it’s output representation.

> In order to address this issue, we introduce an extension to the encoder–decoder model which learns to align and translate jointly.

> The most important distinguishing feature of this approach from the basic encoder–decoder is that it does not attempt to encode a whole input sentence into a single fixed-length vector. Instead, it encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation.

Instead of just using a single vector in the encoded representation, the model splits up different segments into their own encoding vectors, and then individual words can request context from previous phrases.

### Learning to Align and Translate

> The new architecture consists of a bidirectional RNN as an encoder and a decoder that emulates searching through a source sentence during decoding a translation.

**1. Decoder: General Description**

The conditional probability of the next word given the previous words

```math
p(y_i|y_1,...y_{i-1},x) = g(y_{i-1}, s_i, c_i)
```

Where it represents activation of $g$ on a function of the last word $y_{i-1}$, some RNN hidden states $s_i$, and a distinct context vector $c_i$ for each word.

> The context vector $c_i$ depends on a sequence of _annotations_ $(h_1, ..., h_{T_x})$ to which an encoder maps the input sequence. Each annotation $h_i$ contains information about the whole input sequence with a strong focus on the parts surrounding the $i$-th word of the input sequence.

> The context vector $c_i$ is, then, computed as a weighted sum of these annotations $h_i$

```math
c_i = \sum_{j=1}^{T_x} a_{ij}h_j
```

Here, the context vector is a weighted sum of annotations and their respective weights. The annotations then represent the “values” for each previous segment of the sequence, and the weights $a_{ij}$ represent the “relevance” of these values to predicting the current next word.

```math
a_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \\

e_{ij} = a(s_{i-1}, h_j)
```

Here, the model for all $e_{ij}$ use the _alignment model_ $a$ which computes the relevance of the inputs at position $j$ to the output at position $i$, all using a single model. This model is then used to calculate the relative weights by getting the absolute weights for each input and taking the softmax of everything to calculate the relative importance of each input.

> Note than unlike in traditional machine translation, the alignment is not considered to be a latent variable. Instead, the alignment model directly computes a soft alignment, which allows the gradient of the cost function to be back-propagated through.

Providing relevant context from the past for the next word prediction (aligning context) is done implicitly by all functional language models, but this pattern makes it an explicit and optimizable process in this architecture.

> By letting the decoder have an attention mechanism, we relieve the
> encoder from the burden of having to encode all information in the source sentence into a fixed length vector. With this new approach the information can be spread throughout the sequence of annotations, which can be selectively retrieved by the decoder accordingly.

This is the key intuition behind the improvement of attention over previous architectures. Rather than force all relevant context to be compressed into a single fixed length vector by the encoder, the encoder instead creates a number of information bearing vectors for different segments of the previous sentences, and the decoder is able to selectively use relevant information to predict the next word.

**2. Encoder: Bidirectional RNN For Annotating Sequences**

> In the proposed scheme, we would like the annotation of each word to summarize not only the preceding words, but also the following words. Hence, we propose to use a bidirectional RNN. A BiRNN consists of forward and backward RNN’s.

Here, we use two RNN’s - one storing the aggregated contexts by reading the words forward (calculating the _forward hidden states_), and the other the aggregated context coming from reading the words backward (calculating the _backward hidden states_).

These two hidden states are then concatenated to come up with the final annotation for the word.

> In this way, the annotation $h_j$ contains the summaries of both the preceding words and the following words.

### Results

**1. Quantitative Results**

![Screenshot 2024-05-15 at 9.07.07 PM.png](../../images/Screenshot_2024-05-15_at_9.07.07_PM.png)

> One of the motivations behind the proposed approach was the use of a fixed-length context vector in the basic encoder–decoder approach. We conjectured that this limitation may make the basic encoder–decoder approach to underperform with long sentences.

**2. Qualitative Analysis**

![Screenshot 2024-05-15 at 9.12.16 PM.png](../../images/Screenshot_2024-05-15_at_9.12.16_PM.png)

> The proposed approach provides an intuitive way to inspect the (soft-)alignment between the words in a generated translation and those in a source sentence.

> The strength of the soft-alignment, opposed to a hard-alignment, is evident.

Soft-alignment meaning the ability for the model to align across multiple of the different input vectors and take some information from all of them, rather than explicitly mapping input words to output words directly (hard alignment).

> An additional benefit of the soft alignment is that it naturally deals with source and target phrases of different lengths, without requiring a
> counter-intuitive way of mapping some words to or from nowhere

> The proposed model is much better than the conventional at translating long sentences. This is likely due to the fact that [it] does not require encoding a long sentence into a fixed-length vector perfectly, but only accurately encoding the parts of the input sentence that surround a particular word.

### Conclusion

> The conventional approach to neural machine translation, called an encoder–decoder approach, encodes a whole input sentence into a fixed-length vector from which a translation will be decoded. We conjectured that the use of a fixed-length context vector is problematic for translating long sentences.

> In this paper, we proposed a novel architecture that addresses this issue. We extended the basic encoder–decoder by letting a model (soft-)search for a set of input words, or their annotations computed by an encoder, when generating each target word.

> We were able to conclude that the model can correctly align each target word with the relevant words, or their annotations, in the source sentence as it generated a correct translation.

> Perhaps more importantly, the proposed approach achieved a translation performance comparable to the existing phrase-based statistical machine translation. It is a striking result, considering that the proposed architecture, or the whole family of neural machine translation, has only been proposed as recently as this year.
