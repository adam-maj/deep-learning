# Word2Vec

📜 [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)

> We propose two novel model architectures for computing continuous vector representations of words from very large data sets

> We show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.

> Many current NLP systems and techniques treat words as atomic units - there is no notion of similarity between words, as these are represented as indices in a vocabulary.

> This choice has several good reasons - simplicity, robustness and the observation that simple models trained on huge amounts of data outperform complex systems trained on less data.

> However, the simple techniques are at their limits in many tasks.

In many domains, like transcribed speech data, there isn’t much high quality data, so more complex models may be necessary to represent the available.

> Somewhat surprisingly, it was found that similarity of word representations goes beyond simple syntactic regularities.

> It was shown for example that _vector(”King”) - vector(”Man”) + vector(”Woman”)_ results in a vector that is closest to the vector representation of the word _Queen_

> In this paper, we try to maximize accuracy of these vector operations by developing new model architectures that preserve the linear regularities among words.

> We design a new comprehensive test set for measuring both syntactic and semantic regularities, and show that many such regularities can be learned with high accuracy.

This paper tries to focus on creating word representations that preserve syntactic and semantic regularities between different words (the representations contain meeting, and arithmetic can be performed).

> Moreover, we discuss how training time and accuracy depends
> on the dimensionality of the word vectors and on the amount of the training data.

### Model Architectures

> In this paper, we focus on distributed representations of words learned by neural networks.

Neural networks have been shown to be far more effective and computationally efficient than previous approaches like LSA and LDA.

> We will try to maximize the accuracy, while minimizing the computational complexity.

Minimize the the number of parameters needed for the model to fully train while making sure it’s still accurate.

**1. Feedforward Neural Net Language Model (NNLM)**

> The probabilistic feedforward neural network language model has been proposed in. It consists of input, projection, hidden and output layers.

> At the input layer, N previous words are encoded using 1-of-V coding, where V is size of the vocabulary.

> The input layer is then projected to a projection layer P that has dimensionality N × D, using a shared projection matrix.

> Moreover, the hidden layer is used to compute probability distribution over all the words in the
> vocabulary, resulting in an output layer with dimensionality V .

The model consists of:

(1) An input layer of N (often N=10) of the previous words in a one-hot-encoded 1-of-V encoding style for the vocabulary V

(2) There’s a projection layer with dimensionality N × D that maps each vector in the input layer linearly onto a learned representation. This projection layer is the **embedding layer**.

(3) Then, there’s a hidden layer which computes based on the projection layer and has some non-linearities.

(4) Finally, there’s an output layer of dimension V, corresponding with the prediction of the next word.

**2. Recurrent Neural Net Language Model (RNNLM)**

> Recurrent neural network based language model has been proposed to overcome certain limitations of the feedforward NNLM, such as the need to specify the context length, and because theoretically RNNs can efficiently represent more complex patterns than the shallow neural networks.

RNN used for this representation task since they’re theoretically better at word modeling.

> The RNN model does not have a projection layer; only input, hidden and output layer. What is special for this type of model is the recurrent matrix that connects hidden layer to itself, using time-delayed connections.

The embedding layer in this model is not a projection layer but is instead the layer passing information forward in time since it has to form short term memory of the past hidden state.

**3. Parallel Training of Neural Networks**

> To train models on huge data sets, we have implemented several models on top of a large-scale distributed framework called DistBelief, including the feedforward NNLM and the new models proposed in this paper.

### New Log-Linear Models

> In this section, we propose two new model architectures for learning distributed representations of words that try to minimize computational complexity.

> The main observation from the previous section was that most of the complexity is caused by the non-linear hidden layer in the model.

Building models optimized for simplicity. While they may not be able to have as complex representations as neural networks for the task, their computational efficiency is a big benefit.

**1. Continuous Bag-of-Words Model**

> The first proposed architecture is similar to the feedforward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words (not just the projection matrix); thus, all words get projected into the same position (their vectors are averaged).

> We call this architecture a bag-of-words model as the order of words in the history does not influence the projection.

> Furthermore, we also use words from the future; we have obtained the best performance on the task introduced in the next section by building a log-linear classifier with four future and four history words at the input, where the training criterion is to correctly classify the current (middle) word.

In this model, we remove the non-linear layer for the sake of removing complexity, and each word uses the same projection matrix, rather than have it’s own matrix in the projection layer, meaning that there is no information about word position.

**2. Continuous Skip-gram Model**

> The second architecture is similar to CBOW, but instead of predicting the current word based on the context, it tries to maximize classification of a word based on another word in the same sentence.

> More precisely, we use each current word as an input to a log-linear classifier with continuous projection layer, and predict words within a certain range before and after the current word.

> We found that increasing the range improves quality of the resulting word vectors, but it also increases the computational complexity.

This model just tries to predict which words are nearby based on the current word and a certain range.

### Results

> ”What is the word that is similar to small in the same sense as biggest is similar to big?”

> Somewhat surprisingly, these questions can be answered by performing simple algebraic operations with the vector representation of words.

> Finally, we found that when we train high dimensional word vectors on a large amount of data, the resulting vectors can be used to answer very subtle semantic relationships between words, such as a city and the country it belongs to, e.g. France is to Paris as Germany is to Berlin.

**1. Task Description**

> To measure quality of the word vectors, we define a comprehensive test set that contains five types of semantic questions, and nine types of syntactic questions.

![Screenshot 2024-05-15 at 1.49.07 PM.png](../../images/Screenshot_2024-05-15_at_1.49.07_PM.png)

> Question is assumed to be correctly answered only if the closest word to the. vector computed using the above method is exactly the same as the correct word in the question; synonyms are thus counted as mistakes.

> We believe that usefulness of the word vectors for certain applications should be positively correlated with this accuracy metric.

**2. Maximization of Accuracy**

> We have used a Google News corpus for training the word vectors. This corpus contains about 6B tokens. We have restricted the vocabulary size to 1 million most frequent words.

> Increasing amount of training data twice results in about the same increase of computational complexity as increasing vector size twice.

Increasing the vector size of word representations has the same effect as a larger training set.

**3. Comparison of Model Architectures**

The skip-gram model performs best overall.

![Screenshot 2024-05-15 at 1.57.00 PM.png](../../images/Screenshot_2024-05-15_at_1.57.00_PM.png)

### Examples of the Learned Relationships

![Screenshot 2024-05-15 at 1.58.46 PM.png](../../images/Screenshot_2024-05-15_at_1.58.46_PM.png)

> It is also possible to apply the vector operations to solve different tasks. For example, we have observed good accuracy for selecting out-of-the-list words, by computing average vector for a list of words, and finding the most distant word vector.

### Conclusion

> In this paper we studied the quality of vector representations of words derived by various models on a collection of syntactic and semantic language tasks.

> We observed that it is possible to train high quality word vectors using very simple model architectures.

> Using the DistBelief distributed framework, it should be possible to train the CBOW and Skip-gram models even on corpora with one trillion words, for basically unlimited size of the vocabulary.

💬 **Comments**
This paper introduces the intuition for embeddings - it’s actually not a model trained to produce embeddings, but a model trained for a task, and created in a way so that the model is forced to create embeddings somewhere in order to accomplish it’s goal.

Each different model proposed in this paper takes a different approach to language modeling that forces the model to build it’s own embeddings, and the tradeoffs taken are for the sake of computational complexity.

In general, as a model is forced to model certain relationships between words & tokens more accurately to accomplish some task, words that appear together or are contextually similar will be adjusted in the representation space to cluster more closely together, and hopefully, to model relevant syntactic and semantic relationships.

It’s actually very surprising that this happens naturally in the way that the models learn.

The intuition of embeddings is to isolate the available representation space models have to learn so that when they optimize their model of language in their representation space, that space can then be used practically.
