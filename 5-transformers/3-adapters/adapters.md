# Adapters

📜 [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751)

> As an alternative, we propose transfer with adapter modules. Adapter
> modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones.

> Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task.

Adapters allow a base model to fine-tune for a specific task without retraining all of it’s parameters. Instead, they just add on some additional parameters which can get fine-tuned to accomplish the task.

> A high degree of sharing between tasks is particularly useful.

> We propose a transfer learning strategy that yields _compact_ and _extensible_ downstream models.

Compact models don’t require training too many parameters. Extensible models can be continuously updated to learn more tasks.

> Adapter-based tuning requires training two orders of magnitude fewer parameters to fine-tuning, while attaining similar performance.

> Adapters are new modules added between layers of a pre-trained network.

> Consider a function (neural network) with parameters $w: \phi_w(x)$.

> Feature-based transfer composes $\phi_w$ with a new function $\chi_v$ to yield $\chi_v(\phi_w(x))$. Only the new, task-specific parameters, $v$ are then trained.

> Fine-tuning involves adjusting the original parameters, $w$ for each new task, limiting compactness.

> For adapter tuning, $\psi_{w,v_0}(x)$ is defined, where parameters $w$ are copied over from pre-training. […] During training, only $v$ are tuned.

Adapter tuning adds a new set of parameters to be trained, and they work in unison with the original unaltered parameters.

> The key innovation is to design an effective adapter module and its integration with the base model. We propose a simple yet effective, bottleneck architecture.

### Adapter Tuning for NLP

> Our strategy has three key properties: (i) it attains good performance, (ii) it permits training on tasks sequentially, that is, it does not require simultaneous access to all datasets, and (iii) it adds only a small number of additional parameters per task.

> Adapter modules have two main features: a small number of parameters, and a near-identity initialization.

**1. Instantiation for Transformer Networks**

![Screenshot 2024-05-16 at 11.12.45 AM.png](../../images/Screenshot_2024-05-16_at_11.12.45_AM.png)

> The adapter is always applied directly to the output of the sub-layer, after the projection back to the input size, but before adding the skip connection back. The output of the adapter is then passed directly into the following layer normalization.

> To limit the number of parameters, we propose a bottleneck architecture. The adapters first project the original d-dimensional features into a smaller dimension, m, apply a nonlinearity, then project back to d dimensions. […]
>
> The bottleneck dimension, m, provides a simple means to tradeoff performance with parameter efficiency.

### Experiments

> We show that adapters achieve parameter efficient transfer for text tasks.

> The adapter size controls the parameter efficiency, smaller adapters introduce fewer parameters, at a possible cost to performance.

**6. Analysis and Discussion**

> We perform an ablation to determine which adapters are influential.

> First, we observe that removing any single layer’s adapters has only a small impact on performance.

> Adapters on the lower layers have a smaller impact than the higher-layers. […] One intuition is that the lower layers extract lower-level features that are shared among tasks, while the higher layers build features that are unique to different tasks.

> To analyze the impact of the initialization scale on the performance, we
> test standard deviations in the interval. We observe that on both datasets,
> the performance of adapters is robust for standard deviations below $10^{−2}$. However, when the initialization is too large, performance degrades.

Adapters need to be initialized to preserve the identity mapping.
