# DCGAN example

Referenced from [official tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

# GAN

- [Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

GANs are framework for teaching a deep learning model to capture the training data distribution so we can generate new data from that same distribution. They are make of two distinct models, a ***generator*** and a ***discriminator***.

- The job of the generator is to spawn ‘fake’ images that look like the training images. 

- The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator.

The GAN loss function is:

$$
\min\limits_G\max\limits_DV(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_{z}(z)}[\log (1 - D(G(z)))]
$$

# DCGAN

- [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

Generator is comprised of:

- convolutional-transpose layers

- batch norm layer

- ReLU activations

<img src = https://pytorch.org/tutorials/_images/dcgan_generator.png>

Discriminator is make up of:

- strided convolution layers

- batch norm layers

- LeakyReLU activations

<img src = >

# Train step

- Take real data as discriminator input to get error_D_real.

- Take output of generator as discriminator input to get error_D_fake.

- Update discriminator with both gradients.

- Take noise as generator input to get errG.

- Update generator with gradient.