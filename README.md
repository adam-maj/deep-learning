# deep-learning [WIP]

An exploration of the entire history of deep learning from DNNs to GPT-4o.

In the repository, I've included a copy of each critical paper (40+) that got us to where we are today, along with my full explanations of key intuitions/math and a minimal implementation in pytorch for each topic.

On the rest of this page, I provide an overview of everything we can learn from this history, inspired by [_The Lessons of History_](https://www.amazon.com/Lessons-History-Will-Durant/dp/143914995X) by Will & Ariel Durant.

**The project is designed so everyone can get most of the value by reading the overview below.**

Then, people curious to dive into the technical details can explore the rest of the repository via the links in the [resources](#resources) section.

## Table of Contents

- [Overview](#overview-the-lessons-of-history)
  - [1. Constraints](#2-constraints)
    - [1.1. Data](#11-data)
    - [1.2. Parameters](#12-parameters)
    - [1.3. Optimization & Regularization](#13-optimization--regularization)
    - [1.4. Architecture](#14-architecture)
    - [1.5. Compute](#15-compute)
    - [1.6. Compute Efficiency](#16-compute-efficiency)
    - [1.7. Energy](#17-energy)
    - [1.8. Constraints & Leverage](#18-constraints--leverage)
  - [3. Narratives](#3-narratives)
  - [4. Inspiration](#4-inspiration)
  - [5. Intelligence](#5-intelligence)
  - [6. Future](#6-future)
- [Resources](#resources)
- [Papers](#papers)

<br />

# Overview

Digital intelligence is improving quickly, and few have had time to dive deep and fully understand what's going on.

This overview is an effort to highlight the true big picture of how things are progressing, informed by the technical details of individual innovations.

It focuses on two simple questions:

- How did we get here?
- Where are we going?

Through answering these questions, it also explores:

- How is progress made in deep learning?
- Where do the ideas that drive progress in deep learning come from?
- How have our narratives about digital intelligence changed over time?
- What does deep learning teach us about our own intelligence?

<br />

### Context

> Note: If you're already very familiar, skip

[link 3b1b deep learning videos]

Pre-requisite assumptions: you know what deep learning is broadly, how a neural network works, etc.

May cover some topics you're already familiar with, for the sake of explicitness, and because specifics become important in framing.

The goal of deep learning is to create digital intelligence.

We want machines to be able to do complex, economically valuable stuff.

Intelligence = builds model of reality

Builds model of reality = uses data (samples) from reality (distribution) to approximate reality

<br />

# 1. Constraints

Deep learning is about getting good data, and then using that data to model the world.

If we can continue making better models with more useful data, we achieve increasing intelligence.

In practice, there are a few constraints that limit how fast we can accomplish this: data, model size, optimization/regularization, architecture, compute, compute effiency, and energy.

It's impossible to understand where we're going without understanding how we got here. And it's impossible to understand how we got here without deeply understanding each of these constraints, and how they relate to each other.

So let's look at each one in depth.

<br />

## 1.1. Data

![constraint-1-data](./images/readme/constraint-1-data.png)

### True vs. Empirical Distribution

We want our system to effectively understand reality.

More accurately, we want our system to effectively model a data-generating distribution (we'll call this the true distribution).

To do this, the system needs some information about the distribution.

Ideally, it would get _perfect_ information about the distribution. In practice, this is impossible.

So, we settle for a finite amount of information about the true distribution. This information is captured in a collection of individual samples from the true distribution - the dataset.

The samples themselves don't perfectly represent the true distribution. But they do contain some information about it.

The samples provide information about an _approximation_ of the true distribution - which we can call the empirical distribution.

At best, we can expect to learn to model this empirical distribution from the dataset.

<br />

### A Good Approximation

The original goal was to effectively model the true distribution, not the empirical distribution.

So, the empirical distribution needs to be as close as possible to the true distribution. It needs to be "a good approximation."

The empirical distribution becomes a better approximation of the true distribution as we add more data points.

<br />

### Data Quantity & Quality

We can use (simple) information theory to understand how to make the empirical distribution a good approximation of the true distribution.

We need more information about the true distribution in the true distribution - this means more samples (data quantity) where each sample contains a lot of information about the true distribution we want to model (data quality).

So to make our dataset a good approximation of the true distribution, we have two levers: increase quantity of data, and increase quality of data.

<br />

### Breakthrough #1: Public labeled datasets

While deep-learning was getting started, data was collected manually [CNN, LeNet]. This worked at a small scale, but couldn't go very far. Very limited quantity.

The first breakthrough was the creation of large datasets [MNIST, ImageNet]. These enables much larger datasets (relative increase in data quantity), while still maintaining quality.

Instead of each team making their own data, the effort is spread across everyone working on deep learning, and everyone can benefit from these datasets.

This enabled AlexNet to exist, which completely changed deep learning [AlexNet].

But these datasets are unscalable - manual labeling can only get you so far. High data quality, but capped quantity.

<br />

### Breakthrough #2: Network effects create data

The internet exists, creates massive amounts of data through network effects.

These network effects are scalable, and the _only_ reasonable solution to the problem of using large labeled datasets.

They broke past the constraint of limited quantity. But they were unusable - the quality takes a significant drop.

What distribution are these data points sampled from? General human knowledge? Hard to say. And so while there are more data points, those data points are sampled from a more complex distribution - one that's very challenging to model, and unclear how it's useful.

This is unlike labeled datasets like ImageNet, where they were clearly for a task (image classification).

So internet scale datasets appeared to be unusable for a long time.

<br />

### Breakthrough #3: Unlocking the internet

**Transfer learning via pre-training and fine-tuning** enabled us to use internet scale data.

For the first time ever, BERT showed that we could actually make internet-scale datasets useful.

BERT introduced the paradigm of pre-training on the internet (large unlabeled dataset), then fine-tuning on smaller labeled datasets to accomplish a specific task.

> [Graphic] - [Google executive response to BERT](https://x.com/TechEmails/status/1756765277478621620)

This works by acquiring general conceptual knowledge from pre-training on the internet (get the value of high quantity of general data), then repurposing that knowledge for a specific task with a labeled dataset (higher quality data).

This combination enabled us to take advantage of the massive size of unlabeled internet scale datasets, blowing past previous constraints on data by increasing data quantity by many orders of magnitude.

This paradigm continued after with [BERT, RoBERTa, GPT-2, GPT-3]

[LoRA] revealed specifically how this works. The small dataset augments previously ignored dimensions of information - things that the GPT had already learned, but hadn't prioritized. This highlights the symbiotic effect of pre-training and fine-tuning.

Pre-training gives fine-tuning much more leverage with a small amount of data - the fine-tuning dataset does _not_ need to contain a good approximation of the distribution itself. The pre-training phase does.

The fine-tuning phase can have much smaller amount of data, effectively showing the model where to focus in it's large distribution. Ex: "you could respond in many different ways, but focus on responding as if you were an assisstant."

<br />

### Breakthrough #4: Training assistants

InstructGPT was another important breakthrough here. BERT, GPTs, etc. were conceptually interesting, and already making waves

But still didn't appear that useful. InstructGPT is a demonstration of the importance/leverage on data quality, especially having pre-training on a large amount of data.

OpenAI manually created datasets of good responses for an assistant model with human labelers. Many orders of magnitude less data than the original interent dataset.

But it made the difference between GPT-3 launch and ChatGPT.

> [Graphic] - ChatGPT instantly blows up despite GPT-3 existing

<br />

### Can we do better?

Humanoid robots bump the constraint\*\* - The combination of pre-training on internet-scale and fine-tuning on small datasets seems ideal.

Is there anywhere to go from here to push the constraints further, aside from the ever increasing amount of data on the internet?

> [Graphic] - Increasing amount of data generated globally

To keep increasing intelligence, we want our systems to model reality - the distribution we want to model is the laws of reality.

Is the internet the best dataset we can come up with? It could be better in many ways.

Data quality of the internet is a lossy compression of the true distribution we want to model - the internet is data and knowledge about reality filtered through and diluted by the collective intelligence of humanity.

The best case would be to access data about the world directly. With recent humanoid robot arms race, if we see millions of these walking around, they may now have access to start collecting massive amounts of data directly.

> [Graphic] - The robotics arms race: figure, optimus, boston dynamics

### Modeling Data

So the dataset set determines the empirical distribution, which sets the cap on how good of a model we can create. The best model we can create corresponds with how well the empirical distribution approximates the true distribution.

But creating the best model for a dataset is itself a challenging task. With current internet scale datasets, we seem to be far away from making a model that has fully learned the empirical distribution [Link scaling laws chart? Or save for later]

What are the constraints governing how good of a model we can create? These are the remainder of the constraints we'll focus on.

<br />

## 1.2. Parameters

![constraint-2-parameters](./images/readme/constraint-2-parameters.png)

The model itself needs to have enough degrees of freedom to be able to model the empirical distribution.

With modern datasets, the complexity is massive, so more parameters shows no signs of slowing down as a way to increase intelligence.

When the complexity of the empirical distribution is still beyond what the network is capable of modeling, the easiest way to improve the network is to scale up the size.

This means adding more parameters per layer and increasing the depth of the network.

<br />

### Breakthrough #1: Early signs of scaling

Networks naturally scaled over time (combined with other improvements) leading to better results [CNN, LeNet, AlexNet, etc.].

Clearly, size was correlated with better understanding up to a point (until size gets > than distribution complexity).

But at this point, size appears to be _a_ factor, not _the_ factor. Partly, this is because a lot more size was not _needed_ for the types of problems they were solving yet.

<br />

### Breakthrough #2: Scaling laws

GPTs made it clear that for the internet scale dataset, scale is all you need for a while. [GPT-2, GPT-3, GPT-4]

[Scaling laws picture]

The scaling laws show no sign of letting up. This means we are far below the capacity to model the dataset.

Currently, scaling up the size of the model is a clear direction of progress we want to pursur. But this is governed by other constraints (again, the remaining).

<br />

## 1.3. Optimization & Regularization

![constraint-3-optimization-and-regularization](./images/readme/constraint-3-optimization-and-regularization.png)

In practice, you can't just keep scaling up model naively. When scaling up the model, we encounter two classes of problems.

First, when you scale up parameters, and especially when you add more depth, the model may start converging very slowly or stop converging completely. Optimization is used to ensure models can still converge as they grow, especially in depth.

Second, when you scale up parameters past the complexity of the distribution, the model can overfit. Regularization is used to ensure models learn "good representations" to truly model the empirical distribution and not learn noise.

<br />

### Breakthrough #1: Taming gradients

**Vanishing and exploding gradients** - While training deeper networks with many layers, gradients start to get magnified or dissapear to 0, due to the compounding effects of many layers of similar weights.

When gradients vanish and explode like this, large groups of neurons (or even entire layers) can get starved of improvement (coming from backpropagation), making it practically infeasible o texpand beyond a certain point.

Residuals solved this problem. [Residuals]

[Add a graphic of residuals, and add graphics throughout]

By creating a pathway for gradients to flow backwards, they massively increase the capacity of networks to grow in depth [ResNet].

Residuals are still used in most large networks today, and are a key component of transformers. The depth of these models would not work without them. [Transformer]

<br />

### Breakthrough #2: Networks of networks

Expanding models past the complexity of the distribution (which is difficult to predict in practice) can lead to overfitting, which hurts validation loss.

Conceptually, an ideal way to fix this would be to train a network of many different networks and average their effects together. By doing this, different networks are forced to learn the same important representations, and by taking their average, the noise among them cancels out, and the representations remain [Dropout].

In practice, this is computationally impossible - it requires orders of magnitudes more compute.

Dropout accomplished this approach with the same amount of compute. [Dropout]

The idea behind dropout is to split to select a random subset of the network during training my turning a percentage of the neurons off. This creates a subnetwork. Then, training is effectively training an exponential number of these subnetworks, which are all forced to learn useful representations on there own.

Specifically, groups of neurons can't work together, so they have to learn useful representations on their own.

This prevents generalization [Dropout], and again is used everywhere after [Transformer]. This solves regularization. Notably, it increases training time.

<br />

### Breakthrough #3: Taming activations

**Internal covariate shift** - When training even deeper networks, later layers suffer from improving while the outputs of previous layers change, meaning their earlier stages of training are rendered useless.

This again limited how deep networks could get.

Normalization solved this by making sure the input to each neuron stayed within the same input range regardless of previous layers by normalizing activations after each layer. This enables deeper networks again. [BatchNorm, LayerNorm]

<br />

### Breakthrough #4: Momentum

**Adaptive moments** - The initial optimization algorithm for back-propagation, SGD, takes a finite step at each improvement interval. Initially this was fine.

As you start modeling more complex problems, this can get very inefficient. If you have to move in one direction for a long time, continually taking a small step is a waste. If you know you need to go far, you should be taking bigger steps.

Many optimization algorithms started to approach this problem by adding the idea of momentum [AdaGrad, RMSProp].

Adam optimizer combined all these ideas to maintain adaptive moments - keeping track of a running list of past gradients for each parameter, and icnreasing momentum where appropriate, drastically decreasing training time in some scenarios [Adam].

<br />

### The forgotten constraint

All these advancements are used in almost everything today [Transformer], and without them optimization & regularization would be a constraint.

But because of how effective they've been, optimization & regularization are basically "solved" now - you rarely have to think about them when scaling up models now.

This is especially augmented by the fact that we're far from reaching the cap on internet scale dataset complexity, so regularization is hardly a concern.

Despite this, we have to remember that these are still very real constraints, although they don't effect the models in their current state.

<br />

## 1.4. Architecture

![constraint-4-architecture](./images/readme/constraint-4-architecture.png)

Good optimization & regularization enables us to make larger and deeper models.

Are all models of the same size created equal in their capacity to model distributions effectively? Certaintly not.

We want our models to learn “useful representations” that efficiently model the apparent distribution. ie. they learn true information and ignore all the “noise” that comes from random sampling.

If we know a good way the model can capture this “useful information,” and ignore the “noise” using less parameters, we can help it learn more efficiently.

We are telling the model “learn like this.” This is adding _inductive bias_.

Architecture is about more useful representations per group of parameters, which translates to better modeling of apparent distribution, with fewer parameters, meaning more room for parameters, meaning larger effective size again (more useful representations in the mode, more intelligence).

Technically, a DNN with non-linearities can model any distribution, given sufficient scale [Link to justification].

But in practicality, we know there are distributions with so much noise and complexity (like images) that we need some inductive bias beyond basic DNNs to be able to form any useful representations.

<br />

### Breakthrough #1: Learning features unlocks images

**CNN** - The first inductive bias or unique representation really introduced by an architecture [CNN, LeNet, AlexNet], and it's still used today in state of the art generative models [U-Net].

These are about _compression_. They give the model a way to explicitly learn to ignore a lot of noise by only paying attention to specific _features_.

This maps directly to the manual feature engineering that ML engineers did before deep-learning, but with a more deep-learning twist, since it allows the model to learn the best features itself.

<br />

### Breakthrough #2: Memory unlocks sequence modeling

**RNNs -> LSTMs** [RNN, LSTM, Learning to Forget, Encoder-Decoder, Seq2Seq] - First introduces the ability to understand relationships across time & space via memories (because of the introduction of gates).

We enable the model to think about what parts of the data are important to other parts.

The LSTM made the RNN actually viable, and sequence-modeling actually viable. This is the start of the trajectory that led us to where we are now.

However, the LSTM was constrained on time.

<br />

### Breakthrough #3: Attention is all you need

**Attention & Transformers** - [Attention, Transformer] Understand relationships across space without being blocked by time. This enabled parallelization, which increase compute efficiency and model size.

Attention enables every part of an input to learn about every other part of it. Everything is able to enhance the meaning of everything else within context.

The title "Attention Is All You Need" makes the most sense in contrast to previous papers [Encoder-Decoder, Seq2Seq, Attention] that achieved success with RNNs. This paper shows that the _only_ inductive bias you need is attention. This suggests something important about it regarding intelligence.

### Breakthrough #4: Taming randomness

**Generative Models** - By far the most conceptually complex of all the models.

We’ve talked about how we understand samples from complex distributions (with information and noise).

How do we synthesize complex data? It appears we would have to understand not just the information (what we’ve been doing so far), but also the details? The details are extremely complex.

The sum of complex random variables is noise. So we synthesize images by learning to create both features + noise (which adds details).

VAEs create a bottleneck that forces the model to use useful representations. Then add back noise on top of these representations. So we start by creating information, then add noise to it to create our synthesized data.

Diffusion, instead, starts with noise, and learns to add back information to it slowly.

Without these designs, models could never synthesize data.

### Breakthrough #5: Embeddings

**Embeddings** - [Word2Vec, VAEs, CLIP] Force models to learn an interesting representation space with semantic and syntactic meaning.

The classic example of this is that the embeddings allow "King" - "Man" + "Woman" = "Queen"

Embeddings show us the composability of concepts. Transformers use this to soak information into words.

### "Don't touch the architecture"

Combining Models - [DALL E, DALL E 2, Stable Diffusion, etc.] - using U-Net, Transformer, CLIP, VAE, Feed-forward, Diffusion all in one.

Many state of the art models combine pieces from many of the different architectures in different sections to work together to do larger tasks like condition images with text, etc.

GPT-4o is the most obvious example of complete multi-modality, which involves stitching together lots of different types of models.

[8] "Don't touch the architecture" [Clip of Karpathy talking about not changing architectures]

<br />

## 1.5. Compute

![constraint-5-compute](./images/readme/constraint-5-compute.png)

With an efficient architecture and effective optimization & regularization, the remaining constraint on the size of the model is compute.

During training, specifically back-propagation, the gradient for each parameter needs to be computed, and then each parameter updated. This takes compute, which takes time. So with more parameters, there are more computations during back-propagation, which is the limitting step.

So we can train a certain number of parameters per device. And then we need to get more devices. And if there's a limit on how many devices we can use together, we've hit a constraint on compute.

<br />

### Breakthrough #1: Letting compute communicate

**GPU communication** - Using 1 GPU was useful at first, but with larger models, you need to be able to use multiple GPUs at once.

You can't do this by just splitting up the work. The GPUs need to communicate with each other - as little as possible, but still at some points, to synchronize weights/pass data to each other, since this is all one model.

Due to gaming, NVIDIA released GPUs that could communicate with each other, which was used by [AlexNet] to train larger model more effectively.

<br />

### Breakthrough #2: Riding tailwinds

**Gaming tailwinds improve compute** - For most of its time, deep learning was not large enough to justify large companies building dedicated compute for it, especially given how expensive/difficult to execute compute companies are.

Deep learning was lucky that the tailwinds of gaming drove increasing compute quality which deep learning benefitted from on the side. [AlexNet, Transformer, etc.]

<br />

### Breakthrough #3: AI becomes a priority

**AI-first GPUs** - Finally, AI becomes a bet worth taking, NVIDIA releases A100, H100, and now B100 focused on MIG, AI tensor cores, mixed-precision, more FLOPS (smaller floating point numbers).

> [Graphic] - B100

> [Graphic] - Jensen Huang handing OpenAI a GPU

<br />

### Breakthrough #4: BERT & the compute arms race

**The compute arms race** - It wasn't always obvious that compute was going to become a huge constraint at a point in time when the AI narrative was also becoming highly consequential, and garnering large power over capital flows.

This trend was first crafted and visible by OpenAI with [BERT, GPT-2, GPT-3], [SoRA example is another nice example].

They saw this before other people. Sam Altman says "compute is the bottleneck" [Link and correct quote]

Zuck unrelated buys a lot of GPUs because he sees they could be useful, and now they're in a great position because of that. [Link to clip] [Antifragile (redundancy/optionality)]

The arms race really begins - NVIDIA, TSMC, ASML, etc. semi-conductor supply chain prices sky-rocket.

> [Graphic] - Stock prices of all the semi-conductor supply chain companies rising

### Adjusting supply chains

**Adjusting supply chains:** The current constraint on compute is not about people not having funding to buy compute. It's about the compute supply chains not creating enough supply.

Because compute supply chains have a long production cycle (a few months), they rely on predictions. These supply chains did not predict the boom in demand for compute that came from the AI cycle, so they got constrained.

These supply chains will soon adjust to demand, and the constraint on compute will no longer be due to constrained supply chains - instead it will become a resource problem.

### AI ASICs

[7] **AI-accelerator ASICs:** Another wave that may impact compute - in the recent cycles, many companies have raised large amounts of money to built ASICs for AI inference, and now AI training [links? Tenstorrent? etc.]

There's a heuristic that you get 10x boost from chip to FPGA, then another 10x from FPGA to ASIC. This is not exactly accurate in this case since GPU is the benchmark - but the question is, will ASICs really be able to accelerate training (better than NVIDIA, who's already doubling down on AI)?

<br />

## 1.6. Compute Efficiency

![constraint-6-compute-efficiency](./images/readme/constraint-6-compute-efficiency.png)

Making effective use of compute (training parameters most efficiently) is not a gaurantee. This is a software problem and takes active effort and optimization.

Examples like [FlashAttention] show us that there are large compute breakthroughs that accelerate things waiting to happen.

<br />

### Breakthrough #1: CUDA

**CUDA** - The first challenge with compute was just being able to work with GPUs. GPUs require a completely different programming paradigm unfamiliar to most people, and was challenging to get right.

NVIDIA built CUDA for this, a programming pattern native to C which devs were more used to.

We see in [AlexNet] they write their own GPU code manually (and in other places).

<br />

### Breakthrough #2: Kernel libraries

**Kernel Libraries** - People stopped having to write their own kernels as frequently dropping the barrier to entry - solved both by NVIDIAs on kernels and libaries like [TensorFlow], [PyTorch], [JAX] that handle compute side for you.

<br />

### Improvement never ends

**Efficient Implementations** - Even past the point where people write kernels, now that kernel libraries are all available, there are still compute efficiency optimizations lying around. [FlashAttention] notably increased the performance of the Transformer by a huge margin.

> [Graphic] - Sam Altman talking about compute efficiency being good too

<br />

## 1.7. Energy

![constraint-7-energy](./images/readme/constraint-7-energy.png)

Finally, even if we have infinite resources to purchase compute, and more importantly, the supply chain can support any demand, there is still a constraint on compute - energy.

In practice, compute has to be clustered together since it all needs to communicate with each other. Communicating over network is impractical - it has to be much faster - over local InfiniBand.

So compute must all be clustered together in data centers, and the data center will need to have sufficient electricity to keep the compute running.

But with amounts of compute being discussed right now in data centers, the data centers may not actually be able to supply this energy.

> [Graphic] - Microsoft & OpenAI data center

This is because the available energy you can draw from the grid at once place is actually limited to [X GW]. The grid capacity will have to adjust over time to increase this constraint.

> [Graphic] - [Zuck clip about energy](https://www.youtube.com/watch?v=i-o5YbNfmh0)

<br />

## 1.8. Fixed Constraints & Leverage

So we can now consider all these constraints together.

The first constraint is the data (specifically, the quantity and quality of data we have), which determines the empirical distribution. The empirical distribution sets the cap on how well we can model the true distribution based on how well it approximates the true distribution.

The remaining constraints determine how well we can train a model to fit to this empirical distribution. The most immediate is the size of the model, which is actually capped by the amount of compute we have, assuming good architecture, optimization/regularization, and compute efficiency.

And compute can be constrained by energy.

We can conceptually divide these constraints into "hard constraints" and "leverage constraints."

The hard constraints are "data," "compute," "energy." These are much slower to move and are the result of much larger effects - the quality of data determined by the scale of the internet, the amount of compute rate limited by slow moving supply chains and a boom in demand, and energy constraints determined by the slow moving process of increasing available energy from the grid (infrastructural challenge).

Given these hard constraints, we can maximize how effectively we use them via the "leverage constraints" - "size," "architecture," "optimization / regularization," "compute efficiency." These are all easily adjustable (software) and have massive effects [FlashAttention, Residuals, Transformer, GPT-2, etc. etc.].

So you can think of these constraints as offering leverage on the compute/data. With the same compute/data, you can accomplish much more by improving architectures, efficiency, etc.

<br />

# 3. Narratives

Now that we've explored the constraints in detail, we can look back at the history of progress in deep learning through the lens of constraints.

With this perspective, a few key milestones stand out above the rest that completely shifted the paradigm of constraints, and each came with significant shifts in narratives about deep learning.

These narrative shifts alone had a big impact on progress in the industry [Narrative Distillation, Mimetics].

<br />

### Narrative #1: Deep learning works

[1] **AlexNet: deep learning is pointless -> deep learning is viable** - For a long time, machine learning community viewed deep learning as naive. The idea of feature engineering and logic-based ML appealed much more to the narratives we liked about intelligence (logic).

> [Graphic] - AlphaGo, early opinions on deep learning

Papers like [DNN, CNN, LeNet] shifted the narrative of the viability of deep learning. Then, especially [AlexNet] showed people that it should be taken seriously, and really shifted the narrative.

These also demonstrated the first arhictectural breakthrough, and as we saw above, AlexNet appears in innovations in always every single section. They pushed constraints in every direction.

<br />

### Narrative #2: Sequence modeling works

**LSTMs make sequence modeling interesting, and unlock new data** - RNNs were the subject of much effort [RNN] but due to the vanishing/exploding gradients by nature of how RNNs work, they were thought to be completley impractical.

LSTMs made recurrence and sequence modeling viable again [LSTM, Learning to Forget] (results), also creating a large line of work in sequence modeling which eventually led us in the direction of GPTs [Encoder/Decoder, Seq2Seq, Attention].

> [Graphic] - Ilya quote on LSTMs being scalable

Importantly, this shift to sequence modeling as a viable problem now unlocks the first internet scale dataset as a viable data source - all web pages/text on the internet. But, it was initially unclear how this could be used effectively, and the recurrence of the LSTM architecture makes it less efficient training on this data.

<br />

### Narrative #3: Unlocking more data

[3] **Attention is all you need, can process data** - [Attention, Attention is All You Need] Massive architectural breakthrough that has great results, and most importantly, removes the bottleneck of LSTMs to increase parallelization, meaning we can scale up more and use compute more efficiently.

Contrary to what may seem true, this paper alone is not what caused the biggest narrative shifts. It was still unclear how internet scale data could be used.

<br />

### Narrative #4: Scaling laws dominate

[4] **The shift to transfer learning unlocks internet scale data, scaling laws begin** - This may be the biggest recent shift. The pre-training and fine-tuning paradigm introduced by [BERT, RoBERTa, GPT-2, GPT-3, InstructGPT] unlock internet-scale data for the first time ever.

The results from BERT alone cause a narrative shift - these things start to look way more powerful.

> [Graphic] - Google executive thoughts on BERT

Then, GPT-2, GPT-3 start to reveal scaling laws. We finally have a dataset complex enough but useful enough to reveal the nature of scaling laws empirically.

OpenAI noticed this early and takes a bet on it. We see this bet playing out over the papers.

"What's one thing you believe that no one else does" - OpenAI's secret was scaling laws. A few years ago, people used to look down on it. OpenAI turned out to be right.

This narrative in particular highlights the power of narratives in fundraising [Narrative Distillation]. This cycle single-handedly revives SF, billions dumped into AI.

> [Graphic] - Graph of funding in AI, Sam Altman $7T

<br />

# 4. Inspiration

Where do the ideas that have led to breakthroughs in deep learning come from? By looking at the history, we can see a few common sources of inspiration that appear frequently.

<br />

### Neuroscience

**Similar Solutions**

The most common source of apparent inspiration is neuroscience.

The [CNN] is the most obvious architecture directly inspired by neuroscience, as it maps directly to the functinos of the visual system.

Other systems like [LSTM, Attention] appear to draw from neuroscience ideas (memory, attention), although looking at their development, in many ways they may be derived empirically, and then fit onto neuroscience after.

For example, LSTM design is perfectly made to address the vanishing/exploding gradients problem in RNNs, rather than specifically with the goal of modeling memory.

This pattern suggests that rather than taking direct inspiration from neuroscience, deep learning has converged on similar approaches to how nature has built the brain partly through first principles.

**Overfitting**

In early papers, there seems to be an attempt to fit ideas to neuroscience as a justification for their relevance [?, Dropout], which we don't see anymore at all. This reflects on early ideas about deep learning (which have now changed to be less anthropomorphic).

[Dropout] struck me as the most blatant example of this potential overfitting, as they explain "one possible motivation" for dropout as a mapping to animal sexual behavior, despite their prior explanation in the paper of dropout following from a rather logical line of thinking around regularization.

This seems to be an ex-post rationalization of the architecture in an attempt to make it correspond with biology, rather than it actually serving as a source for inspiration (of course, I could be wrong).

<br />

### Linear Algebra & Calculus

Most notably, [back-propagation/DNN] and [LoRA] are directly inspired by the math behind neural networks.

[LoRA] (low-rank adaptation) is directly a manipulation on how models are trained by taking advantage of a feature of linear-algebra (decomposing parameters into lower dimensionality matrices).

<br />

### Physics & Information Theory

Most notably, [VAEs], [Diffusion], [Score-Models], [Flow-Models] all take inspiration from physics - especially Langevin dynamics.

These are systems involving noisy sampling.

<br />

### Nature

In general, all these sources of inspiration are engineering / sources where nature has solved the problem of modeling data in different ways, and we can take ideas from these places.

As mentioned before though, it's very easy to rationalize reasoning for things ex-post that fit into nice narratives.

In reality, the majority of change (though maybe inspired from many places) is heavily grounded in engineering [ie. LayerNorm/BatchNorm, LoRA, Residuals, LSTM from RNN, etc.] - patching errors in things we see, especially using math as the tool for engineering in the context of neural networks.

<br />

# 5. Intelligence

What does this progression tell us about intelligence? I'll try to be very empirical here, because dipping into philosophizing with this topic is very easy if not careful.

One way to view intelligence is a measure of our ability to model complex distributions (about reality), then run active inference using these models to accomplish goals in the world [Free Energy Principle].

This frames [The Cook and the Chef] particularly well - constant update based on the world, constantly improve your own frameworks.

It appears data representing reality (dataset vs. senses) + compute (transistors vs. neurons) + energy (grid vs. food) + scale (params in DNN vs. neurons in brain), we get intelligence

As we’ve defined it, intelligence is certainly emergent from complex systems [Integrated Information Theory] - clearly, there's nothing inherently human about it.

What can architectures teach us about intelligence - maybe more effective representations / parameters is something fundamental. It appears, out of the giant graveyard of stuff tried, that some formats of convolution/compression [visual system], memory [LSTM/forget], attention [attention system] have been tried.

Maybe our brain converged on good ways to model data - the same thing AI is trying to do, and so happens to converge on similar answers over time.

Embeddings mean something in the brain too, and show us something about representations.

And, if intelligence really is just about data + compute + energy, it seems inevitable that digital intelligence will surpass us at some point (to many people this seems inevitable, many disagree, many don’t like it) - this is explored in the next section.

<br />

# 6. Future

We've now reframed the history of progress as a series of break-throughs on the constraints limiting intelligence.

Everything in the past that has contributed to progress has been determined by the constraints discussed above.

And nothing changes in the future - these same constraints will always determine where we're headed, how close we get to AGI, etc.

We've now solved the _theoretical_ problem of AGI for a long time. We know that we need compute, data, and energy and if we keep scaling these, intelligence will increase. This was not obvious until the past 1-2 decades of history (only to some people). It has really only become obvious in the past 3 years - this was a contrarian bet for OpenAI.

The question is now can we solve the _engineering_ problem of AGI. Can we continue pushing on all the constraints to keep increasing intelligence?

It appears optimization/regularization doesn't need to increase too much, and the transformer may be the dominant architecture for a long-time before architecture becomes a plausible constraint again (if ever) [Karpathy saying to keep architecture the same].

We will continue pushing the frontiers on compute efficiency, compute supply chains will adjust, and energy will also adjust slowly (over medium term time horizons). This means we will continue to produce larger and larger parameter models, and intelligence will certainly increase.

How far does this go? It depends how far the scaling laws go. In this context, we see that the scaling laws are not a universal truth. They're a local truth - scale is all we need for now because the data we're training on is far more complex than what current networks are modeling.

This may scale to the point where we achieve intelligence that can improve itself, especially as we unlock access to collect data about the world directly via humanoid robots.

Or, we may cap out at some point and need to figure out new ways to push the data constraint. Either way, given the current trajectory, models by that point will be far more intelligent than they are now, and it's unclear how they'll impact society in the mean time.

<br />

# Resources

> [!IMPORTANT]
>
> Each technical breakthroughs highlighted in this repository is covered in a folder linked below.
>
> In each folder, you'll find a copy of the relevant papers (`.pdf` files), along with my own breakdown of intuitions, math, impact, and my implementation when relevant (all in the `.ipynb` file).

**1. Deep Neural Networks**

- [1.1. DNN](/01-deep-neural-networks/01-dnn/)
- [1.2. CNN](/01-deep-neural-networks/02-cnn/)
- [1.3. AlexNet](/01-deep-neural-networks/03-alex-net/)
- [1.4. UNet](/01-deep-neural-networks/04-u-net/)

**2. Optimization & Regularization**

- [2.1. Weight Decay](/02-optimization-and-regularization/01-weight-decay/)
- [2.2. ReLU](/02-optimization-and-regularization/02-relu/)
- [2.3. Residuals](/02-optimization-and-regularization/03-residuals/)
- [2.4. Dropout](/02-optimization-and-regularization/04-dropout/)
- [2.5. Batch Normalization](/02-optimization-and-regularization/05-batch-norm/)
- [2.6. Layer Normalization](/02-optimization-and-regularization/06-layer-norm/)
- [2.7. GELU](/02-optimization-and-regularization/07-gelu/)
- [2.8. Adam](/02-optimization-and-regularization/08-adam/)

**3. Sequence Modeling**

- [3.1. RNN](/03-sequence-modeling/01-rnn/)
- [3.2. LSTM](/03-sequence-modeling/02-lstm/)
- [3.3. Learning to Forget](/03-sequence-modeling/03-learning-to-forget/)
- [3.4. Word2Vec](/03-sequence-modeling/04-word2vec/)
- [3.5. Encoder-Decoder](/03-sequence-modeling/05-encoder-decoder/)
- [3.6. Seq2Seq](/03-sequence-modeling/06-seq2seq/)
- [3.7. Attention](/03-sequence-modeling/07-attention/)
- [3.8. Mixture of Experts](/03-sequence-modeling/08-mixture-of-experts/)

**4. Transformers**

- [4.1. Transformer](/04-transformers/01-transformer/)
- [4.2. BERT](/04-transformers/02-bert/)
- [4.3. RoBERTa](/04-transformers/03-roberta/)
- [4.4. T5](/04-transformers/04-t5/)
- [4.5. GPT-2](/04-transformers/05-gpt-2/)
- [4.6. GPT-3](/04-transformers/06-gpt-3/)
- [4.7. LoRA](/04-transformers/07-lora/)
- [4.8. InstructGPT](/04-transformers/08-instruct-gpt/)
- [4.9. Vision Transformer](/04-transformers/09-vision-transformer/)
- [4.10. Mixture of Experts Transformer](/04-transformers/10-moe-transformer/)

**5. Image Generation**

- [5.1. GANs](/05-image-generation/01-gan/)
- [5.2. VAEs](/05-image-generation/02-vae/)
- [5.3. Diffusion](/05-image-generation/03-diffusion/)
- [5.4. Stable Diffusion, ControlNet, & SDXL](/05-image-generation/04-stable-diffusion/)
- [5.5. CLIP](/05-image-generation/05-clip/)
- [5.6. DALL E & DALL E 2](/05-image-generation/06-dall-e/)

# Papers

**Deep Neural Networks**

- **DNN** - Learning Internal Representations by Error Propagation (1987), D. E. Rumelhart et al. [[PDF]](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf)
- **CNN** - Backpropagation Applied to Handwritten Zip Code Recognition (1989), Y. Lecun et al. [[PDF]](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
- **LeNet** - Gradient-Based Learning Applied to Document Recognition (1998), Y. Lecun et al. [[PDF]](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **AlexNet** - ImageNet Classification with Deep Convolutional Networks (2012), A. Krizhevsky et al. [[PDF]](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015), O. Ronneberger et al. [[PDF]](https://arxiv.org/abs/1505.04597)

**Optimization & Regularization**

- **Weight Decay** - A Simple Weight Decay Can Improve Generalization (1991), A. Krogh and J. Hertz [[PDF]](https://proceedings.neurips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf)
- **ReLU** - Deep Sparse Rectified Neural Networks (2011), X. Glorot et al. [[PDF]](https://www.researchgate.net/publication/215616967_Deep_Sparse_Rectifier_Neural_Networks)
- **Residuals** - Deep Residual Learning for Image Recognition (2015), K. He et al. [[PDF]](https://arxiv.org/pdf/1512.03385)
- **Dropout** - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014), N. Strivastava et al. [[PDF]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- **BatchNorm** - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Ioffe and C. Szegedy [[PDF]](https://arxiv.org/pdf/1502.03167)
- **LayerNorm** - Layer Normalization (2016), J. Lei Ba et al. [[PDF]](https://arxiv.org/pdf/1607.06450)
- **GELU** - Gaussian Error Linear Units (GELUs) (2016), D. Hendrycks and K. Gimpel [[PDF]](https://arxiv.org/pdf/1606.08415)
- **Adam** - Adam: A Method for Stochastic Optimization (2014), D. P. Kingma and J. Ba [[PDF]](https://arxiv.org/pdf/1412.6980)

**Sequence Modeling**

- **RNN** - A Learning Algorithm for Continually Running Fully Recurrent Neural Networks (1989), R. J. Williams [[PDF]](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)
- **LSTM** - Long-Short Term Memory (1997), S. Hochreiter and J. Schmidhuber [[PDF]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **Learning to Forget** - Learning to Forget: Continual Prediction with LSTM (2000), F. A. Gers et al. [[PDF]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e10f98b86797ebf6c8caea6f54cacbc5a50e8b34)
- **Word2Vec** - Efficient Estimation of Word Representations in Vector Space (2013), T. Mikolov et al. [[PDF]](https://arxiv.org/pdf/1301.3781)
- **Encoder-Decoder** - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014), K. Cho et al. [[PDF]](https://arxiv.org/pdf/1406.1078)
- **Seq2Seq** - Sequence to Sequence Learning with Neural Networks (2014), I. Sutskever et al. [[PDF]](https://arxiv.org/pdf/1409.3215)
- **Attention** - Neural Machine Translation by Jointly Learning to Align and Translate (2014), D. Bahdanau et al. [[PDF]](https://arxiv.org/pdf/1409.0473)
- **Mixture of Experts** - Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017), N. Shazeer et al. [[PDF]](https://arxiv.org/pdf/1701.06538)

**Transformers**

- **Transformer** - Attention Is All You Need (2017), A. Vaswani et al. [[PDF]](https://arxiv.org/pdf/1706.03762)
- **BERT** - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018), J. Devlin et al. [[PDF]](https://arxiv.org/pdf/1810.04805)
- **RoBERTa** - RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019), Y. Liu et al. [[PDF]](https://arxiv.org/pdf/1907.11692)
- **T5** - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019), C. Raffel et al. [[PDF]](https://arxiv.org/pdf/1910.10683)
- **GPT-2** - Language Models are Unsupervised Multitask Learners (2018), A. Radford et al. [[PDF]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **GPT-3** - Language Models are Few-Shot Learners (2020) T. B. Brown et al. [[PDF]](https://arxiv.org/pdf/2005.14165)
- **LoRA -** LoRA: Low-Rank Adaptation of Large Language Models (2021), E. J. Hu et al. [[PDF]](https://arxiv.org/pdf/2106.09685)
- **InstructGPT** - Training language models to follow instructions with human feedback (2022), L. Ouyang et al. [[PDF]](https://arxiv.org/pdf/2203.02155)
- **Vision Transformer** - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020), A. Dosovitskiy et al. [[PDF]](https://arxiv.org/pdf/2010.11929)
- **Mixture of Experts Transformer** - Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models (2023), S. Shen et al. [[PDF]](https://arxiv.org/pdf/2305.14705)

**Generative Models**

- **GAN** - Generative Adversarial Networks (2014), I. J. Goodfellow et al. [[PDF]](https://arxiv.org/pdf/1406.2661)
- **Style GAN** - A Style-Based Generator Architecture for Generative Adversarial Networks (2018), T. Karras et al. [[PDF]](https://arxiv.org/pdf/1812.04948)
- **Style GAN 2** - Analyzing and Improving the Image Quality of StyleGAN (2019), T. Karras et al. [[PDF]](https://arxiv.org/pdf/1912.04958)
- **VAE** - Auto-Encoding Variational Bayes (2013), D. Kingma and M. Welling [[PDF]](https://arxiv.org/pdf/1312.6114)
- **VQ VAE** - Neural Discrete Representation Learning (2017), A. Oord et al. [[PDF]](https://arxiv.org/pdf/1711.00937)
- **VQ VAE 2** - Generating Diverse High-Fidelity Images with VQ-VAE-2 (2019), A. Razavi et al. [[PDF]](https://arxiv.org/pdf/1906.00446)
- **Diffusion** - Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015), J. Sohl-Dickstein et al. [[PDF]](https://arxiv.org/pdf/1503.03585)
- **Denoising Diffusion** - Denoising Diffusion Probabilistic Models (2020), J. Ho. et al. [[PDF]](https://arxiv.org/pdf/2006.11239)
- **Denoising Diffusion 2** - Improved Denoising Diffusion Probabilistic Models (2021), A. Nichol and P. Dhariwal [[PDF]](https://arxiv.org/pdf/2102.09672)
- **Diffusion Beats GANs** - Diffusion Models Beat GANs on Image Synthesis, P. Dhariwal and A. Nichol [[PDF]](https://arxiv.org/pdf/2105.05233)
- **Stable Diffusion** - High-Resolution Image Synthesis with Latent Diffusion Models (2021), R. Rombach et al. [[PDF]](https://arxiv.org/pdf/2112.10752)
- **ControlNet** - Adding Conditional Control to Text-to-Image Diffusion Models (2023), L. Zhang et al. [[PDF]](https://arxiv.org/pdf/2302.05543)
- **SDXL** - SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis (2023), D. Podell et al. [[PDF]](https://arxiv.org/pdf/2307.01952)
- **CLIP** - Learning Transferable Visual Models From Natural Language Supervision (2021), A. Radford et al. [[PDF]](https://arxiv.org/pdf/2103.00020)
- **DALL E** - Zero-Shot Text-to-Image Generation (2021), A. Ramesh et al. [[PDF]](https://arxiv.org/pdf/2102.12092)
- **DALL E 2** - Hierarchical Text-Conditional Image Generation with CLIP Latents (2022), A. Ramesh et al. [[PDF]](https://arxiv.org/pdf/2204.06125)
