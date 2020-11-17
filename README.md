## Table of contents

- [1. Background](#1-background)
  - [1.1 Micro scale: Film grain vs. Digital noise](#11-micro-scale--film-grain-vs-digital-noise)
  - [1.2 Macro scale: Color rendition](#12-macro-scale--color-rendition)
- [2. Collecting data](#2-collecting-data)
- [3. Training and Testing](#3-training-and-testing)
- [4. A note on adversarial examples](#4-a-note-on-adversarial-examples)
- [5. References](#references)
- [6. License](#license)


## Analog Classifier

![](https://github.com/juanpablolebon/analog-classifier/blob/main/Example%20Images/Yosemite.png)

A neural network that discerns images taken with an analog camera from images taken with a digital camera

## 1. Background
The “film look” is a term thrown around fairly often by photography and cinema enthusiasts. It is employed as a rejection of the comparatively clean and accurate image rendition of digital cameras, and an embrace of the stylized (and so less accurate) interpretation yielded by film stocks.
Let’s briefly look at the two major differences between these mediums, to hopefully give some intuition as to why the approach taken later on to train the model was picked and why it worked.

## 1.1 Micro scale: Film grain vs. Digital noise

You don’t have to look at an entire image to notice qualities that give away the kind of camera they were taken with: one important difference lies in the texture. Analog cameras capture light that comes in through the lens on film stock, while digital cameras capture it on a digital sensor.

Film stocks contain halide crystals which react to light and reduce to metallic silver. These clumps of crystals are what yield the image while the sections of the film stock that do not contain these crystals create the grain. The details of the chemistry behind this process aren’t entirely relevant here, what matters is the fact that film grain is more of an optical illusion than anything else, since it is merely the sections of stock that do not contain reduced silver. These won’t display any color and will therefore be blank. 

Digital noise, meanwhile, is the result of electronic circuits injecting unwanted signals into the sensor recording the image. This means that it has a markedly different look to film grain since it is born out of a completely different process. 

![](https://i.stack.imgur.com/e7lqN.png)

Film grain is widely accepted to be more desirable than digital noise: it’s smooth and evenly distributed, while digital noise means sharp, pixel-level distortions in the color.

## 1.2 Macro scale: Color rendition

For all the debate there is about digital vs film, a lot of the time these two mediums end up looking fairly similar. The look people associate with film, at least in photography, is more about making a picture look like _underexposed_ film:

![](https://github.com/juanpablolebon/analog-classifier/blob/main/Example%20Images/Underexposure%20Example.png)

Film does not handle underexposure that well - it clearly distorts colors significantly after just a handful of stops of underexposure. It’s fortunate, though, that a lot of people actually like this distortion, so much so that it’s the look they associate with film. Properly exposed film still has a distinct look, though it’s more subtle and therefore not trivial to discern from a digital photo.

For this project I focused specifically on human faces. While this may seem like an overly specific result, or a cop-out to avoid tackling other categories of photographs (landscapes, architecture), the fact that I achieved good results on human faces is a good indicator that this approach generalizes rather well to other categories. This is due to the fact that digital cameras and (most) film stocks tend to render skin tones just as, if not more, similarly than they render other colors frequently found in the world of photography. In a lot of cases, in fact, it can be quite tricky to tell which is which when presented with a side-by-side comparison.

![](https://github.com/juanpablolebon/analog-classifier/blob/main/Example%20Images/Skin%20rendition%20example.png)

It would not be too difficult to translate this method to other types of photography and build a more general tool, it would only involve training models on different subjects with the same approach.

## 2. Collecting data

The dataset involved around 100 portraits taken on each type of camera for a total of slightly over 200. The film portraits were taken from the film photography community on Reddit, [r/analog](https://reddit.com/r/analog). The digital portraits were taken from the [Flickr-Faces-HQ Dataset](https://github.com/NVlabs/ffhq-dataset).

Despite planning to build a model involving images as inputs, I knew I didn’t want to use a convolutional neural network because I specifically needed to ignore the placement of colors on an image. After all, a medium isn’t dictated by what the medium photographs. Associating chunks of light blue on the top of an image with a specific medium would be undesired behavior. 

Even after making the decision to train only on faces, a similar logic applied. The shapes and expressions of the faces didn’t matter. I knew, from what was explained above, that the way neighboring pixels interacted with and compared to each other was relevant, since grain/noise and colors on pixels that are close together are what give each medium its look. An approach where I used single pixels as data, then, wouldn’t work as well, even if it may seem sensible to take this approach on the hypothesis that certain colors tend to only show up on one of the mediums and not the other. It’s possible that a recurrent neural network would work well with single pixels since it could theoretically have some sense of context for neighboring pixels, though it’s not a path I explored this time.

I opted, then, to populate the database with an algorithm I made myself. I wrote an algorithm that scanned images taking NxN grids of pixels and mapping them to one row of 3*[N^2] cells (3 cells for each pixel due to me splitting RGB values into their own cell, instead of keeping each cell as a three-dimensional vector). A value of N=6 worked very well, though it’s possible other values may work even better.

## 3. Training and Testing

Once I had the dataset ready I wrote a sequential model with Keras, using the Adam optimizer and a learning rate of 0.001, both of which worked best out of the options I tried. About 10 epochs were enough for an accuracy above 90%.

Subsequent predictions were done in a slightly manual way: I wrote a function that takes a specified number of NxN grids and maps them to dataframe rows via the same process used when gathering the data, and then makes predictions using Keras for each of these rows. If a given row is predicted to come from a digital image, I increase a “result” variable by one. I decrease this variable by one if this is not the case. Luckily the model is accurate enough that working with a very low number of rows is viable; the amount of times where the majority of rows were predicted inaccurately was very low.

Testing the model on a different batch of 100 faces yielded an accuracy of over 95%, indicating that the model did not overfit.

## 4. A note on adversarial examples

Something interesting about this model was that it was seemingly invulnerable to adversarial examples devised via the Fast Gradient Sign Method.

Just for a bit of context, FGSM is a tool used to construct data that a model will misclassify. Ordinary neural networks update their parameters by taking them in the opposite direction of the gradient of the network’s associated Cost function (since the gradient of a function gives the direction of fastest growth at a given point). FGSM, then, does the opposite. It takes advantage of neural networks’ linearity and creates noise that, when added into a regular piece of data, changes its values in the direction that the gradient determined would increase the Cost function the most. In other words it creates perturbed data that simultaneously features the most subtle change possible from the original as well as the highest damage to a network’s ability to classify it correctly. The magnitude of this change is dictated by a choice of a multiplier epsilon, a positive real number usually much smaller than 1.

![](https://www.tensorflow.org/tutorials/generative/images/adversarial_example.png)

Constructing a normal dataframe of grids mapped to rows (as usual), and adding a perturbed dataframe created via FGSM with an epsilon of smaller than 1 barely made a dent on the accuracy and confidence of this model. Moreover, ridiculous epsilon values of 1000 were not enough to change the model’s prediction at times. At worst it decreased the model’s confidence in its prediction from, say, 90% to 30%.


## 5. References

[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)



## 6. License
[MIT](https://choosealicense.com/licenses/mit/)
