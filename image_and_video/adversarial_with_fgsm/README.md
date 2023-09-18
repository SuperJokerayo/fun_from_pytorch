# An Adversarial Example

Referenced from [official tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html).

## Preliminary

There are several kinds of assumptions of the attacker’s knowledge, two of which are: **white-box** and **black-box**.

- A white-box attack assumes the attacker has full knowledge and access to the model, including architecture, inputs, outputs, and weights. 

- A black-box attack assumes the attacker only has access to the inputs and outputs of the model, and knows nothing about the underlying architecture or weights. 

There are also several types of goals, including **misclassification** and **source/target misclassification**.

- A goal of misclassification means the adversary only wants the output classification to be wrong but does not care what the new classification is.

- A source/target misclassification means the adversary wants to alter an image that is originally of a specific source class so that it is classified as a specific target class.

In this case, the FGSM attack is a ***white-box*** attack with the goal of ***misclassification***. With this background information, we can now discuss the attack in detail.

The idea of **FGSM (Fast Gradient Sign Attack)** is simple, rather than working to minimize the loss by adjusting the weights based on the backpropagated gradients, the attack adjusts the input data to maximize the loss based on the same backpropagated gradients. 


<img src = https://pytorch.org/tutorials/_images/fgsm_panda_image.png>

FGSM creates perturbed image as:

$$
perturbed\_ image=image+epsilon\cdot sign(data\_ grad)=x+\epsilon\cdot sign(\nabla_x J(\theta,\mathbf{x},y))
$$

## No Free Lunch

As epsilon increases the test accuracy decreases **BUT** the perturbations become more easily perceptible. In reality, there is a tradeoff between accuracy degradation and perceptibility that an attacker must consider. 

## Relavent Papers

- [Adversarial Attacks and Defences Competition](https://arxiv.org/pdf/1804.00097.pdf)

- [Audio Adversarial Examples: Targeted Attacks on Speech-to-Text](https://arxiv.org/pdf/1804.00097.pdf)
