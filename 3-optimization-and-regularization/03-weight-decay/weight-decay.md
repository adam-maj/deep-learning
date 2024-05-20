# Weight Decay

📜 [A Simple Weight Decay Can Improve Generalization](https://proceedings.neurips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf)

> It is proven that a weight decay has two effects in a linear network. First, it suppresses any irrelevant components of the weight vector by choosing the smallest vector that solves the learning problem. Second, if the size is chosen right, a weight decay can suppress some of the effects of static noise on the targets, which improves generalization quite a lot.

> Bad generalization occurs if the information [in the training set] does not match the complexity [of the network].

> A different way to constrain a network and thus decrease its complexity, is to limit the growth of the weights through some kind of weight decay.

> [This] can be realized by adding a term to the cost function that penalizes large weights.

$E(w) = E_0(w) + \frac{1}{2} \lambda \sum_i w_i^2$

> The aim of the learning is not only to learn the examples, but to learn the underlying function that produces the targets for the learning process.

> A small weight decay will pick out the point in the valley with the smallest norm among all the points in the valley.

> In general it can not be proven that picking that solution is the best strategy. But, at least from a philosophical point of view, it seems sensible, because it is (in a loose sense) the solution with the smallest complexity-the one that Ockham would probably have chosen.

Applying Ockham’s razor.

> The value of a weight decay is more evident if there are small errors in the targets.

Smaller weights means less ability to fit noise.
