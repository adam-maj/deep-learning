# LeNet

📜 [Back-propagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)

> The ability of learning networks to generalize can be greatly enhanced by providing constraints from the task domain.

> Previous work performed on recognizing simple digit images showed that good generalization on complex tasks can be obtained by designing a network architecture that contains a certain amount of a priori knowledge about the task.

Adding priors and inductive bias is how to design architectures to make them better for specific tasks.

> The basic design principle is to reduce the number of free parameters in the network as much as possible without overly reducing its computational power.

> Application of this principle increases the probability of correct generalization because it results in a specialized network architecture that has a reduced entropy, and a reduced Vapnik-Chervonenkis dimensionality.

Reducing parameters (and providing similar computation power through architecture) makes the model more generalized and prevents it from overfitting to noise.

The images are cut into 16x16 images of numbers, with varying grayscale pixel values form -1 to 1.

### Network Design

**1. Input and Output**

> All the connections in the network are adaptive, although heavily constrained, and are trained using back-propagation.

Unlike previous implementations, no layers (including feature layers) are manual. They all are optimized via back-propagation.

Here is an early intuition that manually picking features is almost always worse than letting compute optimize the parameters for us.

> The input of the network is a 16 by 16 normalized image. The output is composed of 10 units (one per class) and uses place coding.

We see that the output is to predict which number has been detected.

**2. Feature Maps and Weight Sharing**

> Classical work in visual pattern recognition has demonstrated the advantage of extracting local features and combining them to form higher order features.

> Such knowledge can be easily built into the network by forcing the hidden units to combine only local sources of information.

We can build a feature detection system into the neural network.

> Distinctive features of an object can appear at various locations on the input image. Therefore it seems judicious to have a set of feature detectors that can detect a particular instance of a feature anywhere on the input plane.

> Since the _precise_ location of a feature is not relevant to the classification, we can afford to lose some position information in the process. Nevertheless, _approximate_ position information must be preserved, to allow the next levels to detect higher order, more complex features.

We want a sufficient level of invariance. Features a few pixels a part should be treated similarly, but we still want to distinguish between a specific feature in the top-left vs. bottom-right, especially to be able to combine smaller features into larger ones.

So, we want to create invariance across small changes, but preserve information about large differences - effectively compressing the starting image down to only useful information.

> The detection of a particular feature at any location on the input can be easily done using the “weight sharing” technique.

> Weight sharing not only greatly reduces the number of free parameters in the network but also can express information about the geometry and topology of the task.

We use weight sharing to allow individual **kernels** to be created by the neural network that can recognize specific features across the entire network. These kernels are “shared” across all locations in the input.

> In our case, the first hidden layer is composed of several planes that we call _feature maps_. All units in a plane share the same set of weights, thereby detecting the same feature at different locations.

Feature maps use the same set of weights and multiply them by different input values from different locations to extract the same feature across the image.

> Since the exact position of the feature is not important, the feature maps need not have as many units as the input.

**3. Network Architecture**

![Screenshot 2024-05-09 at 12.49.39 PM.png](../../images/Screenshot_2024-05-09_at_12.49.39_PM.png)

The network has three hidden layers H1, H2, and H3.

> Connections entering H1 and H2 are local and are heavily constrained.

H1 has 12 groups of 64 units arranged as 12 independent 8x8 feature maps. Each unit in a feature map has a 5x5 pixel input from the input plane.

> For units in layer H1 that are one unit apart, their receptive fields (in the input layer) are two pixels apart. Thus, the input image is _undersampled_ and some position information is eliminated.

Units of the same type of feature share weights across the image, whereas units across different features obviously don’t. Also, units don’t share their biases (even within a feature)

### Results

> Some kernels synthesized by the network can be interpreted as feature detectors remarkably similar to those found to exist in biological vision systems and/or designed into previous artificial character recognizers, such as spatial derivative estimators or off-center/on-surround type feature detectors.

Features that the network automatically converges to detecting match some feature times similar to how the brain operates or that were previously designed by humans.

> The first several stages of processing in our previous system involved convolutions in which the coefficients had been laboriously hand designed.

> In the present system, the first two layers of the network are constrained to be convolutional, but the system automatically learns the coefficients that make up the kernels.

> This “constrained back-propagation” is the key to success of the present system: it not only builds in shift-invariance, but vastly reduces the entropy, the Vapnik-Cervonenkis dimensionality, and the number of free parameters, thereby proportionately reducing the amount of training data required to achieve a given level of generalization performance.

Adding these constraints/priors into the network architecture significantly reduces the computational complexity of the problem, increases generalization, and completely eliminates the need to have humans laboriously craft individual features (and performs better).

### Conclusion

> We have successfully applied back-propagation learning to a large, real-world task.

> This work points out the necessity of having flexible “network design” software tools that ease the design of complex, specialized network architectures.
