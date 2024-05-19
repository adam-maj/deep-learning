# InstructGPT

📜 [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)

> Making language models bigger does not inherently make them better at following a user’s intent.

> In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback.

> In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation.

> Our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

> We want language models to be _helpful_, _honest_, and _harmless_.

> We focus on fine-tuning approaches to aligning language models. Specifically, we use reinforcement learning from human feedback (RLHF) to fine-tune GPT-3 to follow a broad class of written instructions. This technique uses human preferences as a reward signal to fine-tune our models.

Using RLHF to align the model with human preferences via fine-tuning.

> We first hire a team of 40 contractors to label our data, based on their performance on a screening test.

This is still an extremely manual and human involved process.

The procedure is:
(1) collect a dataset of human-written desired output behaviors and some labeler written prompts and use this to train GPT-3.
(2) next, collect a dataset of human-labeled comparisons between outputs, then train a reward model (RM) to predict which outputs labelers prefer.
(3) then use the RM as a reward function to maximize reward for the model using PPO.

> This procedure aligns the behavior of GPT-3 to the stated preferences of a specific group of people (mostly our labelers and researchers), rather than any broader notion of “human values”. We call the resulting models InstructGPT.

### Methods and experimental details

> Step 1: Collect demonstration data, and train a supervised policy
> Step 2: Collect comparison data, and train a reward model
> Step 3: Optimize a policy against the reward model using PPO

### Results

**1. Results on the API distribution**

> Labelers significantly prefer InstructGPT outputs over outputs from GPT-3

> Our models generalize to the preferences of “held-out” labelers that did not produce any training data.

> Public NLP datasets are not reflective of how our language models are used.

**2. Results on public NLP datasets**

> InstructGPT models show improvements in truthfulness over GPT-3

> We can minimize performance regressions on public NLP datasets by modifying our fine-tuning procedure.

**3. Qualitative Results**

> InstructGPT models show promising generalization to instructions outside of the RLHF fine-tuning distribution.

> InstructGPT still makes simple mistakes.

### Discussion

**1. Implications for alignment research**

> The cost of increasing model alignment is modest relative to pre-training.

> We’ve seen some evidence that InstructGPT generalizes ‘following instructions’ to settings that we don’t supervise it in.

> We were able to mitigate most of the performance degradations introduced by our fine-tuning.

> We’ve validated alignment techniques from research in the real world.
