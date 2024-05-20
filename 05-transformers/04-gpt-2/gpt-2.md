# GPT-2

📜 [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

> We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText.

Language models start to learn and be able to complete tasks that typically required fine-tuning and supervised learning in an unsupervised way if they are just given enough data to train on.

> Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still under-fits WebText.

> Our suspicion is that the prevalence of single task training on single domain datasets is a major contributor to the lack of generalization observed in current systems.

> It will be very difficult to continue to scale the creation of datasets and the design of objectives to the degree that may be required to brute force our way there with current techniques. This motivates exploring additional setups for performing multitask learning.

Multitask learning is promising for approaching general intelligence in language models, but it is expensive to create labeled datasets for it.

> We demonstrate language models can perform down-stream tasks in a zero-shot setting – without any parameter or architecture modification.

### Approach

> Language modeling is also able to, in principle, learn the tasks of without the need for explicit supervision of which symbols are the outputs to be predicted.

> The internet contains a vast amount of information that is passively available without the need for interactive communication. Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement.

Here’s the hypothesis of OpenAI that leads to all their scaling laws research. The intuition is that the internet already has a ton of data and that providing the model with this data will make it learn more than people expect.

**1. Training Dataset**

> Our approach motivates building as large and diverse a dataset as possible in order to collect natural language demonstrations of tasks in as varied of domains and contexts as possible.
> A promising source of diverse and nearly unlimited text is web scrapes such as Common Crawl.

> Instead, we created a new web scrape which emphasizes document quality. To do this we only scraped web pages which have been curated/filtered by humans.

We see improvements in taste on the dataset (determined by humans) improving the quality of the model. And the broader trend of improving the quality of web scrapes and the model size to improve the model itself.

**2. Input Representation**

> We prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

### Experiments

> Our largest model, which we call GPT-2, has over an order of magnitude more parameters than GPT.

![Screenshot 2024-05-16 at 11.59.59 AM.png](../../images/Screenshot_2024-05-16_at_11.59.59_AM.png)

### Discussion

> Much research has been dedicated to learning, understanding, and critically evaluating the representations of both supervised and unsupervised pre-training methods. Our results suggest that unsupervised task learning is an additional promising area of research to explore.

> While zero-shot performance establishes a baseline of the potential performance of GPT-2 on many tasks, it is not clear where the ceiling is with fine-tuning.

### Conclusion

> When a large language model is trained on a sufficiently large and diverse dataset it is able to perform well across many domains and datasets.
