# Clustering-oriented representation learning with Attractive-Repulsive loss
Code for my [AAAI 2019 paper](https://arxiv.org/abs/1812.07627) in the 
Network Interpretability for Deep Learning Workshop.

This repository includes the following:
- The split-up AGNews dataset into 8 topics (see directory `agnews`);
- Implementations of Gaussian-COREL and Cosine-COREL in high- and low-level ways for ease of integration.

## COREL Implementations
Here, you can the different ways for implementing COREL models, depending on your use-cases.

- Direction 0 (the high-level API): pass your model (which does NOT have an output layer) into the constructor for a  `CORELWrapper` class, such that you can simply feed forward any input to get predictions, then using the function `get_loss_function()` to get exactly the correct loss that you will need.

- Direction 1 (the low-level API): use the loss functions, prediction functions, and attractive-repulsive helpers directly as you see fit.

See `example.py` for a simple, straightforward example of how to do option 0 (recommended).

---
If you have any questions, please feel free to email me at kiankd@gmail.com.

Best,
Kian 

