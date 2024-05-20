# Seq2Seq

📜 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)

> In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure.

> Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector.

This uses the encoder-decoder architecture with LSTMs for sequence-to-sequence tasks.

> Given that translations tend to be paraphrases of the source sentences, the translation objective encourages the LSTM to find sentence representations that capture their meaning, as sentences with similar meanings are close to each other while different sentences meanings will be far. A qualitative evaluation supports this claim, showing that our model is aware of word order and is fairly invariant to the active and passive voice.

### The Model

The three main changes from the default LSTM formulation.

> First, we used two different LSTMs: one for the input sequence and another for the output sequence, because doing so increases the number model parameters at negligible computational cost and makes it natural to train the LSTM on multiple language pairs simultaneously.

> Second, we found that deep LSTMs significantly outperformed shallow LSTMs, so we chose an LSTM with four layers.

> Third, we found it extremely valuable to reverse the order of the words of the input sentence.

### Experiments

**1. Dataset Details**

> We used the WMT’14 English to French dataset. We trained our models on a subset of 12M sentences consisting of 348M French words and 304M English words.

**2. Decoding and Rescoring**

> Once training is complete, we produce translation by finding the most likely translation according to the LSTM

```math
\hat{T} = \arg \max_T p(T|S)
```

> We search for the most likely translation using a simple left-to-right beam search decoder which maintains a small number $B$ of partial hypotheses, where a partial hypothesis is a prefix of some translation.

> As soon as the “<EOS>” symbol is appended to a hypothesis, it is removed from the beam and is added to the set of complete hypotheses.

Beam search is used to traverse the tree of sequence predictions to create a number of hypotheses, and then the available hypotheses that get created can get scored and selected from.

**3. Reversing the Source Sentences**

> We discovered that the LSTM learns much better when the source sentences are reversed.

> While we do not have a complete explanation to this phenomenon, we believe that it is caused by the introduction of many short term dependencies to the dataset.

As with all attempts at interpretability in deep learning research, there are results that are empirically definitive, but the explanations are not clear. Attempts at rationalization are made post-hoc.

> Normally, when we concatenate a source sentence with a target sentence, each word in the source sentence is far from its corresponding. word in the target sentence. As a result, the problem has a large “minimal time lag.”

By reversing the sentence order, this “minimal time lag” is reduced, which could impact backpropagation.

> Backpropagation has an easier time “establishing communication” between the source sentence and the target sentence, which in turn results in substantially improved overall performance.

**6. Experimental Results**

> While the decoded translations of the LSTM ensemble do not outperform the best WMT’14 system, it is the first time that a pure neural translation system outperforms a phrase-based SMT baseline on a large scale MT task by a sizeable margin, despite its inability to handle out-of-vocabulary words.

![Screenshot 2024-05-15 at 5.25.40 PM.png](../../images/Screenshot_2024-05-15_at_5.25.40_PM.png)

**8. Model Analysis**

> One of the attractive features of our model is its ability to turn a sequence of words into a vector of fixed dimensionality.

![Screenshot 2024-05-15 at 5.21.38 PM.png](../../images/Screenshot_2024-05-15_at_5.21.38_PM.png)

![Screenshot 2024-05-15 at 5.22.18 PM.png](../../images/Screenshot_2024-05-15_at_5.22.18_PM.png)

### Conclusion

> In this work, we showed that a large deep LSTM, that has a limited vocabulary and that makes almost no assumption about problem structure can outperform a standard SMT-based system whose vocabulary is unlimited on a large-scale MT task.

> We were surprised by the extent of the improvement obtained by reversing the words in the source sentences. We conclude that it is important to find a problem encoding that has the greatest number of short term dependencies, as they make the learning problem much simpler.

> We were also surprised by the ability of the LSTM to correctly translate very long sentences.

> Most importantly, we demonstrated that a simple, straightforward and a relatively unoptimized approach can outperform an SMT system, so further work will likely lead to even greater translation accuracies.
