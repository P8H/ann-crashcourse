Artificial Neural Networks Crash-Course
===

## from the practical point of view

###### using [deeplearning4j](https://github.com/deeplearning4j)

---

# MultiLayerConfiguration
<!-- page_number: true -->
Configuration example
```java
MultiLayerConfiguration nn_conf = new NeuralNetConfiguration
	.seed(123)
    .iterations(10)
    .learningRate(0.0001)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(Updater.NESTEROVS)
    .list(layers)
    .pretrain(true)
    .backprop(true)
    .build();
```
[Documentation](http://deeplearning4j.org/neuralnet-configuration)

---
# Configuration
## Core Params

##### _`.seed(123)`_
Random number generator seed. Used for example for the initial values for the nodes.
##### _`.iterations(1 ... 10)`_
How many times the NN will be optimized by pretrain & back propagation. Not exactly the same as [epoch](http://deeplearning4j.org/glossary.html#a-nameepochepoch-vs-iterationa). Greater than 1 only at full-batch training. Default: 1.
##### _`.learningRate(1e-1 ... 1e-6)`_
Defines the impact of a back propagation step. Low values causes slow learning, high values may lead to miss the optimum.

---

## Optimization Algorithms
Calculates the error by the gradient
##### _`.optimizationAlgo(OptimizationAlgorithm.*)`_

- LINE_GRADIENT_DESCENT
- CONJUGATE_GRADIENT
- [HESSIAN_FREE](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf) _(2nd-order method)_
- [LBFGS](https://cs231n.github.io/neural-networks-3/#second-order-methods) _(2nd-order method)_
- STOCHASTIC_GRADIENT_DESCENT _(default)_

---

## Updater
The mechanism for weight updates in backpropagation. Adjust often the learning rate.
##### _`.updater(`[`Updater.*`](http://deeplearning4j.org/glossary.html)`)`_
- SGD _(stochastic gradient descent, default)_
- ADAM
- ADADELTA
- NESTEROVS _(used `.momentum(*)` parameter, recommended)_
- ADAGRAD
- RMSPROP
- NONE
- CUSTOM


---


## Training
##### _`.pretrain(true)`_
Activates pretrain for all layer which are pretrain able e.g the layer types RBM and [Autoencoder](http://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/feedforward/autoencoder/AutoEncoder.html).

##### _`.backprop(true)`_
Activates back propgation which updates the weights of the network after every `.fit(data)` based on the evaluated error with the OptimizationAlgorithm and the Updater.

---


## Layer options
##### _`.activation(String)`_
Activation function for the neurons. Diagrams [[1]](http://rasbt.github.io/mlxtend/user_guide/general_concepts/activation-functions/) [[2]](https://en.wikipedia.org/wiki/Activation_function)
- "relu" [0, 1] (rectified linear, most popular for DNN)
- "leakyrelu"
- "tanh" (-1, 1)
- "sigmoid" (0, 1)? (default)
- "softmax"
- "hardtanh"
- "maxout"
- "softsign" (-1, 1)
- "softplus" (0, infinity)


---


## Layer options

##### _`.weightInit(WeightInit.*)`_ 
- Distribution: Sample weights from a distribution based on shape of input
- Normalized: Normalize sample weights
- Size: Sample weights from bound uniform distribution using shape for min and max
- Uniform: Sample weights from bound uniform distribution
- VI: Sample weights from variance normalized initialization
- Zeros
- Xavier (default)
- RELU
###### _Description from source code from deeplearning4j_
---



## Recommended configurations
|Hidden Layer|Output Layer|WeightInit|
|:-:|:-:|:-|
|relu/leakyrelu|softmax (classification) |RELU
|tanh|*linear*|XAVIER

---


## Layer options
##### _`.lossFunction(LossFunctions.LossFunction.RMSE_XENT)`_
Will be used for pretraining and the OutputLayer
* MSE: (Mean Squared Error, Linear Regression)
* EXPLL: (Exponential log likelihood, Poisson Regression)
* XENT (Cross Entropy, Binary Classification)
* MCXENT (Multiclass Cross Entropy, Classification)
* RMSE_XENT (RMSE Cross Entropy)
* SQUARED_LOSS
* RECONSTRUCTION_CROSSENTROPY (default)
* NEGATIVELOGLIKELIHOOD
* CUSTOM

---

## Generalization options
##### _.dropOut(double)_
##### _.l2(double)_
##### _.setUseRegularization(boolean)_

*coming soon*


---

# Layer types

---

# RBM Layer
## Restricted Boltzmann Machine
```java
layera[0] = new RBM.Builder()
.nIn(100)
.nOut(150)
.lossFunction(LossFunctions.LossFunction.RMSE_XENT)
.visibleUnit(VisibleUnit.BINARY)
.hiddenUnit(HiddenUnit.BINARY)
.build()
```
[More about RBM](http://deeplearning4j.org/restrictedboltzmannmachine.html)

---

## RBM Layer

##### _`.visibleUnit(VisibleUnit.*).hiddenUnit(HiddenUnit.*)`_
- LINEAR (visible only)
- BINARY (default)
- GAUSSIAN
- SOFTMAX
- RECTIFIED (rectified linear units, hidden only)

---

## Recommended configurations
|visible unit|hidden unit|note|stability
|:-:|:-:|:-|:-|
|BINARY|BINARY|default| ++
|SOFTMAX|BINARY||+
|RECTIFIED|BINARY|| 0
|GAUSSIAN|BINARY||-
|GAUSSIAN|RECTIFIED|for continuous data|-
|RECTIFIED|RECTIFIED||--
|GAUSSIAN|GAUSSIAN||---
*less stable configurations needs lower learning rates*





---
http://deeplearning4j.org/glossary.html
http://deeplearning4j.org/troubleshootingneuralnets
http://www.dkriesel.com/science/neural_networks




