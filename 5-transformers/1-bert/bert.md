# BERT

📜 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.

> BERT is designed to pre-train deep bidirectional representations from
> unlabeled text by jointly conditioning on both
> left and right context in all layers.

> As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, […] without substantial task-specific architecture modifications.

BERT is built specifically for fine-tuning. It makes it easy to train a single base model and then use it to create a number of task specific models.

> BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks.

> There are two existing strategies for applying pre-trained language representations to downstream tasks: _feature-based_ and _fine-tuning._

> We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches.

> The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer.

> Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

The main problem of the original transformer design is that the left-to-right architecture during training means that words learn to soak up context from words on their left, but not on their right, whereas in understanding sentences, soaking up context from all directions is critical.

> BERT alleviates the previously mentioned unidirectionality constraint by using a “masked language model” (MLM) pre-training objective.

> The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on it’s context.

Using this MLM objective, words can learn to absorb context from all other words around them, making the Transformer bidirectional.

> We demonstrate the importance of bidirectional pre-training for language representations.

> We show that pre-trained representations reduce the need for many heavily-engineered task specific architectures.

### **Related Work**

**1. Unsupervised Feature-based Approaches**

> Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch.

**2. Unsupervised Fine-tuning Approaches**

> Sentence or document encoders which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task.

**3. Transfer Learning from Supervised Data**

> There has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference and machine translation.

### BERT

> During pre-training, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks.

> A distinctive feature of BERT is its unified architecture across different tasks.

BERT is built specifically for easy fine-tuning for a number of different tasks, and the architecture of the model stays exactly the same after fine-tuning.

**1. Pre-training BERT**

> In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens.

> In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task.

**Next sentence prediction (NSP)** is an essential addition of BERT. The model is trained with some classification problems - and it’s initial token for every predicted sequence is the `[CLS]` token, meant to classify if the two sentences it was fed follow each other or not.

Adding this task into the model forces the model to learn whether two sentences are actually related or not, rather than just assuming that the text it’s fed is all correctly related.

**2. Fine-tuning BERT**

> For each task, we simply plug in the task specific inputs and outputs into BERT and fine-tune all the parameters end-to-end.

> Compared to pre-training, fine-tuning is relatively inexpensive.

### Ablation Studies

**1. Effect of Pre-training Tasks**

![Screenshot 2024-05-16 at 10.24.26 AM.png](../../images/Screenshot_2024-05-16_at_10.24.26_AM.png)

**2. Effect of Model Size**

> It has long been known that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling, which is demonstrated by the LM perplexity of held-out training data.

![Screenshot 2024-05-16 at 10.28.04 AM.png](../../images/Screenshot_2024-05-16_at_10.28.04_AM.png)

### Conclusion

> Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems.

> Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.
