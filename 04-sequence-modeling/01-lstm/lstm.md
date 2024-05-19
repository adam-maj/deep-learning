# LSTM

📜 [Long Short-Term Memory](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf)

> Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error back-flow.

> [We] address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM).

> LSTM is local in space and time; its computational complexity per time step and weight is O(1).

> In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster.

> LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.

LSTM is a far more efficient and effective learning algorithm than all previous approaches to RNNs.

> In principle, recurrent networks can use their feedback connections to store representations of recent input events in the form of activations (short-term memory, as opposed to long-term memory embodied by slowly changing weights).

> With conventional backpropagation through time (BPTT) or real-time recurrent learning (RTRL), error signals flowing backward in time tend to (1) blow up or (2) vanish; the temporal evolution of the back-propagated error exponentially depends on the size of the weights.

The central problem - gradients explode or vanish as they get back-propagated through time because the same weights are contributing multiple times to the output through each time-step.

> This article presents long short-term memory (LSTM), a novel recurrent network architecture in conjunction with an appropriate gradient-based learning algorithm.

> LSTM is designed to overcome these error back-flow problems.

The LSTM is an architecture built specifically to fix the gradient back-flow errors (vanish/exploding gradients) faced by previous RNN architectures.

> It can learn to bridge time intervals in excess of 1000 steps even in case of noisy, incompressible input sequences, without loss of short-time-lag capabilities.

### Previous Work

This section is a good indicator of how much work happens at the frontiers of research and on problems before a correct and effective solution is discovered. We see 9 different attempts at long-term memory modeling to improve RNNs before the LSTM emerges.

### Constant Error Backpropagation

**1. Exponentially Decaying Error**

We see from analyzing the error flow occurring at a unit $u$ at time step $t$ propagated back into time for $q$ time steps to a unit $v$ (indicated by the quantity $\frac{\partial{\vartheta_v(t-q)}}{\partial{\vartheta_u(t)}})$, we get the following term:

```math
\prod_{m=1}^q f'_{l_m}(\textrm{net}_{l_m}(t - m))w_{l_ml_{m-1}}
```

And we can see that if

```math
|f'_{l_m}(\textrm{net}_{l_m}(t - m))w_{l_ml_{m-1}}| > 1.0
```

then the largest product increase exponentially with $q$, meaning that the error blows up. Meanwhile, if the quantity is less than 1.0, the error converges to 0.

**2. Constant Error Flow: A Naive Approach**

> To avoid vanishing error signals, how can we achieve constant error flow through a single unit j with a single connection to itself?

We want a way to back-propagate error through previous time-steps without the errors vanishing or exploding.

We know (confirmed from the RNN paper) that we can define the back-propagated error signal to unit j as:

```math
\vartheta_j(t) = f'_j(\textrm{net}_j(t)) \sum_i w_{ij} \vartheta_i(t+1)
```

Where the error pathway flows backward via the recursive $\vartheta(t+1)$ pipeline all the way from the most recent time-step where the original error originates.

Thus, we can simplify a unit j’s local error back-flow if it has only a single connection to itself in the previous time-step (for the sake of simplicity) as

```math
\vartheta_j(t) = f'_j(\textrm{net}_j(t)) \vartheta_j(t+1) w_{jj}
```

And to make sure that we have constant error flow backward through this pathway without vanishing/exploding gradients, we know that $f_j(x) = x$ as the activation function and $w_{jj}$ as the weight value are satisfactory.

There are two obvious challenges with this setup.

First is **input weight conflict**.

We can examine this neuron and add a single input weight to it $w_{ji}$ to demonstrate some additional challenges.

> Assume that the total error can be reduced by switching on unit j in response to a certain input and keeping it active for a long time.

In other words, we want this cell to respond to an input by storing it in the cell $j$, and then use this cell $j$ to persist that value across future time-steps, meaning it should ignore future values to $i$.

> $w_{ji}$ will often receive conflicting weight update signals during this time (recall that j is linear).

In our example, on one time-step, the loss may suggest that $w_{ji}$ should be increased (too allow $i$ to propagate to $j$) as a function of the input, since the neuron is a linear mapping, whereas $w_{ji}$ should be at 0 for the remaining time-steps to prevent intervention. Thus, these needs each contribute separate, inconsistent updates to $w_{ji}$

The next is **output weight conflict**.

> Assume j is switched on and currently stores some previous input. For simplicity, let us focus on a single additional outgoing weight $w_{kj}$. The same $w_{kj}$ has to be used for both retrieving $j$’s content at certain times and preventing $j$ from disturbing $k$ at other times.

This is the same problem, mirrored to outputs. Sometimes we want this pathway to contribute it’s value, whereas at other times we want the value to be able to be ignored.

> As the time lag increases, stored information must be protected against perturbation for longer and longer periods, and, especially in advanced stages of learning, more and more already correct outputs also require protection against perturbation.

This reflects the primary challenge of storing and maintaining long-term memories. Over time, you’re faced with the challenge of storing correct long-term memories, and then ensuring that important memories aren’t distributed.

This is the reasoning behind the LSTM.

### The Concept of Long Short-Term Memory

**1. Memory Cells and Gate Units**

In order to construct an architecture that allows for constant error flow, while mitigating the errors discussed above, the CEC is extended with additional features.

> A multiplicative input gate unit is introduced to protect the memory contents stored in $j$ from perturbation by irrelevant inputs, and a multiplicative output gate unit is introduced to protect other units from perturbation by currently irrelevant memory contents stored in $j$.

> The resulting, more complex unit is called a memory cell.

> Each memory cell is built around a central linear unit with a fixed self-connection.

This linear unit represents the CEC which persists the cell state (long-term memory).

![Screenshot 2024-05-14 at 2.26.28 PM.png](../../images/Screenshot_2024-05-14_at_2.26.28_PM.png)

Here, most importantly, we have the linear long-term error pathway through the middle with the recurrence, where we have

```math
s_{c_j}(0) = 0 \\
s_{c_j}(t) = s_{c_j}(t-1) + y^{\textrm{in}_j}(t)g(\textrm{net}_{c_j}(t))
```

Indicating that the linear CEC pathway is just the sum of it’s value in the previous time-step, plus the value computed by the multiple of the activation of the current cells input, and the activation of the previous cells output (short-term memory).

Intuitively, the activation of the cells input can be thought of as the “value” to add to long-term memory, and the activation of the previous cell (short-term memory) can be thought of as multiplying by it’s “relevance” to determine how much to add to the long-term memory pathway.

Similarly, the output can be calculated as

```math
y^{c_j}(t) = y^{\textrm{out}_j}(t)h(s_{c_j}(t))
```

Which can be thought of as the multiple of the output gate activation to this cell which is the “relevance” of the long-term memory in this case, as well as the actual activation on the current long-term memory pathway, which is the “value” of long-term memory to pull from as the output of this cell.

In-terms of actual weight and bias connections to other cells, there are a few relevant distinctions.

First, the actual “input” value to the cell, contributing the input gate is a function of the weights and outputs of other cells (coming from their short-term memory outputs) in the previous time-step.

```math
y^{\textrm{in}_j}(t) = f_{\textrm{in}_j}(\textrm{net}_{\textrm{in}_j}(t)) \\
\textrm{net}_{\textrm{in}_j} = \sum_u w_{\textrm{in}_ju}y^u(t-1)
```

And similarly, the actual “output” value contributing to the “relevance” of the long-term memory pathway for the cell is influenced by the weights and outputs of other cells in the previous time-step

```math
y^{\textrm{out}_j}(t) = f_{\textrm{out}_j}(\textrm{net}_{\textrm{out}_j}(t)) \\
\textrm{net}_{\textrm{out}_j} = \sum_u w_{\textrm{out}_ju}y^u(t-1)
```

And finally, that the actual previous short-term memory coming from this cell specifically in the previous time step is considered via

```math
\textrm{net}_{c_j}(t) = \sum_u w_{c_ju} y^u (t-1)
```

**2. Why Gate Units?**

> In other words, the net can use $\textrm{in}_j$ to decide when to keep or override information in memory cell $c_j$ and $\textrm{out}_j$ to decide when to access memory cell $c_j$ and when to prevent other units from being perturbed by $c_j$.

> Error signals trapped within a memory cell’s CEC cannot change, but
> different error signals flowing into the cell (at different times) via its output gate may get superimposed.

Sometime’s the error signal at the most recent time-step will suggest that the CEC should have held an increased memory value, sometimes decreased. These will be different at different time-steps.

The gate values are what determines how much different time-steps should affect the gates of an individual cell, and how much the gradients should be back-propagated in time.

For example, a single cell may only be active given the presence of a specific type of concept, and hold memory about that concept in context - then in periods where the concept is presented, the in gate is turned on, then when it needs to be used as context, the output gate’s are on.

Hence, only in these type steps where the input or output gates should be active are the gradients from the CEC back-propagated through to actually update the weights from this cell, otherwise the gates close out the gradients.

> The output gate will have to learn which errors to trap in its CEC by appropriately scaling them.

When an error is particularly relevant to a cell, the output gate needs to be scaling the CEC pathway to have a high impact on the cells output.

In reverse, this means that the gradient will be back-propagated through the CEC pathway since it had a high influence, allowing previous time-steps to get the error.

So in the context of back-propagation, the output gate can be thought of as “trapping” the error within the CEC pathway _only_ when the error is one relevant to the learned function of a cell.

> The input gate will have to learn when to release errors, again by appropriately scaling them.

Conversely, the input gate captures the errors coming through the CEC pathway (as trapped by an output cell in the current or later time-step), and “releases” the errors to be back-propagated through the parameters of the cell, only when relevant (when there is something important to contribute to the long-term memory pathway).

> Essentially the multiplicative gate units open and close access to constant error flow through CEC.

This is the core intuition behind the gates of the LSTMs, and is critical. This explains how back-propagation using the continuous error pathway is distributed properly to cells that need it, but not where it’s irrelevant.

**4. Memory Cell Blocks**

> S memory cells sharing the same input gate and the same output gate form a structure called a memory cell block of size S. Memory cell blocks facilitate information storage

Memory cell blocks still only have two gates (which means they will learn about a single representation and it’s input/output relevance) but they are able to store more complex memories than just a single CEC pathway is capable of.

**5. Learning**

> To ensure non-decaying error backpropagation through internal states of memory cells, as with truncated BPTT, errors arriving at memory cell net inputs do not get propagated back further in time (although they do serve to change the incoming weights).

This answer my question about how the same vanishing/exploding gradients don’t get back-propagated through time just like in RNNs. We clip the gradients and explicitly prevent back-propagation in time through the short-term memory pathway.

> Only within memory cells, are errors propagated back through previous internal states $s_{c_j}$.

Back-propagation through time only occurs through the CEC pathway.

**6. Computation Complexity**

> Only the derivatives $\partial{s_{c_j}}/\partial{w_{il}}$ need to be stored and updated. Hence the LSTM algorithm is very efficient, with an excellent update complexity of $O(W)$

> Unlike full BPTT, however, LSTM is local in space and time: there
> is no need to store activation values observed during sequence processing in a stack with potentially unlimited size.

Back-propagation in LSTMs is local in space and time - units don’t need to store activations of all previous units since the gradient flows just through the CEC pathway.

**7. Abuse Problem and Solutions**

> In the beginning of the learning phase, error reduction may be possible without storing information over time. The network will thus tend to abuse memory cells, for example, as bias cells.

In the early phase, there isn’t much use for long-term memory, so the network can abuse the utility of the long-term memory cells.

**8. Internal State Drift and Remedies**

> If memory cell $c_j$’s inputs are mostly positive or mostly negative, then its internal state $s_j$ will tend to drift away over time.

### Experiments

> Which tasks are appropriate to demonstrate the quality of a novel longtime-lag algorithm? First, minimal time lags between relevant input signals and corresponding teacher signals must be long for all training sequences.

> Recently we discovered that many long-time-lag tasks used in previous
> work can be solved more quickly by simple random weight guessing than by the proposed algorithms.

> All our experiments involve long minimal time lags.

For the sake of testing, experiments use examples where information must be retained for a long time and then used later on.

**6. Experiment 6: Temporal Order**

> In this subsection, LSTM solves other difficult (but artificial) tasks that have never been solved by previous recurrent net algorithms. The experiment shows that LSTM is able to extract information conveyed by the temporal order of widely separated inputs.

This experiment really shows the superiority of LSTMs over RNNs in being able to solve problems that RNNs never could.

### Discussion

**1. Limitations of LSTM**

> Each memory cell block needs two addition units. In comparison to standard recurrent nets, however, this does not increase the number of weights by more than a factor of 9.

> LSTM does not have any problems with the notion of recency that go beyond those of other approaches. All gradient-based approaches, however, suffer from a practical inability to count discrete time steps precisely.

**2. Advantages of LSTM**

> The constant error backpropagation within memory cells results in LSTM’s ability to bridge very long time lags.

> For long-time-lag problems such as those discussed in this article,
> LSTM can handle noise, distributed representations, and continuous
> values.

> There appears to be no need for parameter fine tuning. LSTM works
> well over a broad range of parameters such as learning rate, input gate
> bias, and output gate bias.

> The LSTM algorithm’s update complexity per weight and time step is
> essentially that of BPTT, namely, $O(1)$.

### Conclusion

> Each memory cell’s internal architecture guarantees constant error flow
> within its CEC, provided that truncated backpropagation cuts off error flow trying to leak out of memory cells. This represents the basis for bridging very long time lags.

> Two gate units learn to open and close access to error flow within each memory cell’s CEC. The multiplicative input gate affords protection of the CEC from perturbation by irrelevant inputs. Similarly, the multiplicative output gate protects other units from perturbation by currently irrelevant memory contents.

💬 **Comments**
In some ways, it seems like the LSTM is also about context and attention, just like the Transformer, but it’s just far less efficient in training due to it’s lack of parallelization.

Having individual cell blocks let’s the network learn about the relevance between different types of important information across time.

Really, what a memory cell is learning to do over time is recognize some important piece of information, store it over time and don’t modify it while it is irrelevant/nothing contributes useful information to it, and then release that information when it becomes relevant. In many ways, this is like attention and context where words can store their values over time, waiting to contribute their values to other relevant words.
