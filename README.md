# FaceNet

## Table of contents

* [Table of contents](#table-of-contents)
* [Introduction](#introduction)
* [Installation](#installation)
* [How to use](#how-to-use)
  + [*Load reference embeddings*](#load-reference-embeddings)
  + [*Predict new faces*](#predict-new-faces)
* [Additional](#additional)
* [Citations](#citations)

## Introduction

FaceNet_pytorch is an implementation of the FaceNet convolutional neural network in the PyTorch framework. FaceNet is a facial recognition technique that has become very popular in recent years due to its high accuracy and robustness.

FaceNet\_pytorch uses the technique of embeddings to represent faces as high-dimensional vectors, which allows them to be compared using similarity measures such as Euclidean distance. Additionally, it uses a convolutional neural network that learns to extract high-level features from face images in its last layers, allowing for a robust and invariant representation under different lighting, pose, and facial expression conditions. In summary, FaceNet_pytorch uses embeddings and a convolutional neural network to achieve high accuracy in facial recognition.

In this repo, what was built is a framework to abstract from all the embeddings logic. Different methods are provided to compare face images.

## Installation

All you need is Python and to install the requirements.
```bash
cd your_project_path/face_detection
pip install -r requirements.txt
```

## How to use

The operation is simple, it is necessary to load the embeddings (of the faces) that your application will use as reference, and then, given new face images, the model will identify if that face matches any of the references. If you want to know how this works, read the additional section.

#### Load reference embeddings

First, we instantiate the model.
```python
from face_detection import FaceDetector

model = FaceDetector()
```

Now, to load the new embeddings, the model has the method calculate\_reference\_embeddings. This method receives a list of tuples, where each tuple contains an image and a label that represents the same. You could have the same label for different embeddings, which would mean that all those embeddings represent the same face.

One way to create this data structure would be the following:
```python
import os
from PIL import Image

images_path = "your_path_images"
x_reff = []

for filename in os.listdir(images_path):
    if filename.endswith((".webp", ".png", ".jpg")):
        image = Image.open(os.path.join(images_path, filename)).convert('RGB')
        x_reff.append((filename.split(".")[0], image)) # append new tuple
```

> Note: The model works only with 3-channel (RGB) images, if you have 4-channel images then you must first convert them to 3 channels.

Once we have the list of tuples, we can calculate the embeddings for the model:

```python
model.calculate_reference_embeddings(x_reff)
```

Finally, If we want to save/load the embeddings in memory, we can use:
```python
path_to_save_embeddings = #your parth to save embeddings
path_to_load_embeddings = #your path to load embeddings

model.save_reference_embeddings(path_to_save_embeddings)
model.load_reference_embbedings(path_to_load_embeddings)
```

By default, the embeddings are saved and loaded in "your\_path\_project/face_detection/embeddings"

### Predict new faces

Now we can inquire about new faces to the model:
```python
new_image = #load a new image in format RGB ()
model.get_id_by_image(new_image)
```

## Additional

FaceNet is a convolutional neural network. These networks are good at obtaining hierarchical high-level representations, meaning that as we move to the last layers, the obtained features become increasingly higher-level, unlike the first layers. The interesting thing about these representations is that they are robust and invariant under different lighting conditions, pose, facial expressions, etc. This is useful to determine, for example, if two face images belong to the same person.

Now, to determine if two faces belong to the same person, instead of comparing the images pixel by pixel (which is highly unfeasible because even a slight variation in lighting would make the images completely different), high-level representations of convolutional networks (embeddings) are used to make such a comparison.

Now, if we have a CNN trained to recognize faces, it will calculate good embeddings in its last layers, so similar faces will have similar embeddings, which could be relatively close in their latent space, which is not what we are interested in. Therefore, what FaceNet does, is to train precisely for embeddings of different faces to be as far away from each other as possible in their latent space, while embeddings of the same face should be close. In other words, the network is trained so that the distance between embeddings is as large as possible, in order to have greater certainty when comparing them.

The metric used to compare embeddings is the Euclidean distance. Embeddings of the same face will have a small Euclidean distance, while two embeddings of different faces will have more distant embeddings.

If you want more details, I invite you to read the original paper on siamese networks, which I have included in the citations.

## Citations

Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition [http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf]. En Computer Vision and Pattern Recognition (CVPR).  







