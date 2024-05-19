# Learning to Forget

📜 [Learning to Forget: Continual Prediction with LSTM](https://www.researchgate.net/profile/Felix-Gers/publication/12292425_Learning_to_Forget_Continual_Prediction_with_LSTM/links/5759414608ae9a9c954e84c5/Learning-to-Forget-Continual-Prediction-with-LSTM.pdf)

> We identify a weakness of LSTM networks processing continual input streams that are not _a priori_ segmented into subsequences with explicitly marked ends at which the network's internal state could be reset.

> Without resets, the state may grow indefinitely and eventually cause the network to break down.

As they were initially designed, the CEC pathway of an LSTM cell would infinitely increase, causing it to become useless after a while.

> Our remedy is a novel, adaptive \forget gate" that enables an LSTM cell to learn to reset itself at appropriate times, thus releasing internal resources.

Instead, by adding a forget gate, the LSTM can recycle it’s resource over time. For example, it can learn to have a trigger to “store” some context with the input gate, a trigger to “use” the context with some output gate, and then subsequently “reset” to be able to pick up new context.

> We review illustrative benchmark problems on which standard LSTM outperforms other RNN algorithms. All algorithms (including LSTM) fail to solve continual versions of these problems. LSTM with forget gates, however, easily solves them in an elegant way.

If you extend the problems to require continuous inputs and thinking without obvious stopping points, even LSTMs fail to solve the problems - but after adding forget gates, everything works!

> Standard RNNs fail to learn in the presence of time lags greater than 5-10 discrete time steps between relevant input events and target signals.

An empirical measure of how much long-term memory standard RNNs were actually capable of handling.

> The vanishing error problem casts doubt on whether standard RNNs can indeed exhibit significant practical advantages over time window-based feedforward networks.

Cool to see doubts cast on things that worked in hindsight. People didn’t believe RNNs could be useful until the LSTM was introduced.

> In this paper, however, we will show that even LSTM fails to learn to correctly process certain very long or continual time series. […] The problem is that a continual input stream eventually may cause the internal values of the cells to grow without bound, even if the repetitive
> nature of the problem suggests they should be reset occasionally.

Some problems motivate that a cell should be able to reset itself.

> We recognize that any training procedure for RNNs which is powerful enough to span long time lags must also address the issue of forgetting in short term memory (unit activations). We know of no other
> current training method for RNNs which is suciently powerful to have encountered this problem.

They also point out that forgetting is an essential job of any RNN, and so the indication that LSTMs have finally hit this problem is an indication of how far ahead they are.

### Standard LSTM

**1. Limits of standard LSTM**

> [LSTM’s ability to store data across time lags] can contribute to a weakness in some situations: the cell states $s_c$ often tend to grow linearly during the presentation of a time series.

Importantly, if the long-term memory grows linearly over time, this will cause saturation of the output squashing function $h$, causing the gradient to vanish and preventing learning over time.

> Of course, we might try to “teacher force” the internal states $s_c$ by resetting them once a new training sequence starts. But this requires an external teacher that knows how to segment the input stream into training subsequences. We are precisely interested, however, in those situations where there is no a priori knowledge of this kind.

They want to create a solution that doesn’t require any previous knowledge, and can learn on it’s own when to discard irrelevant information.

### Solution: Forget Gates

**1. Forward Pass of Extended LSTM with Forget Gates**

> Our solution to the problem above are adaptive \forget gates" designed to learn to reset memory blocks once their contents are out of date and hence useless.

The forget gate has a similar activation as the other gates.

```math
y^{\varphi_j}(t) = f_{\varphi_j}(\textrm{net}_{\varphi_j}(t)) \\
\textrm{net}_{\varphi_j}(t) = \sum_m w_{\varphi_jm}y^m(t-1)
```

> We use the logistic sigmoid as squashing function, hence the forget gate’s activation $y^\varphi$ ranges between 0 and 1.

> Bias weights for LSTM gates are initialized with negative values for input and output gates, positive values for forget gates.

Given the activation functions, this means that the input and output gates of each cell will be initialized to never activate (storing and using no context), whereas the forget gate will be set to always remember (as the activation will be almost 1.0) and will have to learn what to forget.

> It will not explicitly forget anything until it has learned to forget.

**2. Backward Pass of Extended LSTM with Forget Gates**

> Truncation means that all errors are cut o once they leak out of a memory cell or gate, although they do serve to change the incoming weights. The effect is that the CECs are the only part of the system through which errors can ow back forever. This makes LSTM's updates efficient without significantly affecting learning power: error flow outside of cells tends to decay exponentially anyway.

Just like the other gates, error flow past forget gates is not back-propagated and is truncating, allowing the CEC pathway to be the only back-propagated path for gradients to flow.

**3. Complexity**

> To calculate the computational complexity of extended LSTM we take into account that weights to input gates and forget gates cause more expensive updates than others, because each such weight directly affects all the cells in its memory block.

> Extended LSTM is local in space and time, just like standard LSTM.

### Experiments

**4. Analysis of CERG Results**

We can see the constant growth of typical unmodified LSTM states below.

![Screenshot 2024-05-14 at 4.01.45 PM.png](../../images/Screenshot_2024-05-14_at_4.01.45_PM.png)

Whereas with the forget gate, we can see the activation of the forget gate and the subsequent resets and adjustments of internal states, making memory cells usable again.

![Screenshot 2024-05-14 at 4.01.39 PM.png](../../images/Screenshot_2024-05-14_at_4.01.39_PM.png)

> Common to all memory blocks is that they learned to reset themselves in an appropriate fashion.

**6. Continual Noisy Temporal Order Problem**

> Can standard LSTM solve problems which extended LSTM cannot? We tested extended LSTM on one of the most difficult nonlinear long time lag tasks ever solved by an RNN: “Noisy Temporal Order”

> Now we take the next obvious step and transform the NTO into a continual problem that does require forgetting.

They create the CNTO problem that tests the ability of the LSTM to forget.

### Conclusion

> Continual input streams generally require occasional resets of the stream-processing network.

> Partial resets are also desirable for tasks with hierarchical decomposition.

> Since typical real-world input streams are not a priori decomposed into training subsequences, and since typical sequential tasks are not a priori decomposed into appropriate subproblems, RNNs should be able to learn to achieve appropriate decompositions.

> LSTM's gates (and forget gates in particular) provide an example of local, efficient information processing through adaptive, multiplicative units, which are, due to their biological plausibility, attracting attention in the field of neuroscience

Interesting how the appeal to neuroscience seems to be something that researches wanted to make their work align with, especially in older papers.
