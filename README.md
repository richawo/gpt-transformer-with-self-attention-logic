
# GPT Transformer with Self-Attention Logic

This repository contains Python code for implementing a GPT (Generative Pre-trained Transformer) transformer model with self-attention logic. The code is based on Andrej Karpathy's tutorial on creating a transformer from scratch.

I've included extra notes to help explain how self-attention works in a transformer architecture.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Usage](#usage)

## Introduction

The GPT transformer model is a powerful natural language processing model known for its ability to generate human-like text. This repository provides an implementation of the model along with self-attention logic.

## Setup

Before running the code, make sure you have the following dependencies installed:

- Python 3
- PyTorch
- Jupyter Labs

You can install the required libraries using the following command:

```shell
pip install torch jupyterlab
```

## Data

The model is trained on Shakespeare's works, which are provided in the `input.txt` file. You can download the data using the provided script:

```shell
!wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt
```

## Model

The GPT transformer model is implemented in the `BigramLanguageModel` class. It uses a token embedding table and self-attention logic to generate text.

## Training

To train the model, an AdamW optimizer is used with a learning rate of 1e-3. Training is done for a specified number of iterations.

## Usage

You can use the trained model to generate text by providing a starting context. Here's an example of how to use it:

```python
from BigramLanguageModel import BigramLanguageModel

# Initialize the model
model = BigramLanguageModel()

# Generate text with a given context
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=1000)

# Decode and print the generated text
decoded_text = decode(generated_text[0].tolist())
print(decoded_text)
```

Feel free to explore and experiment with the code to generate text based on your desired context.