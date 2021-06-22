# swin-transformer-keras

This repository contains the `tensorflow.keras` implementation of the Swin Transformer and its demonstrations under benchmark datasets.

# Notebooks

* Swin Transformer example with the MNIST dataset [[link](https://github.com/yingkaisha/swin_transformer_keras/blob/main/Swin_Transformer_MNIST.ipynb)].

# Dependencies

* TensorFlow 2.5.0, Keras 2.5.0, Numpy 1.19.5.

# Overview

Swin Transformers (Liu et al., 2021) are Transformer-based computer vision models that feature shifted window-based self-attention. Compared to other vision transformer variants that compute embedded patches (i.e., tokens) globally, the Swin Transformer computes a subset of tokens through non-overlapping windows that are alternatively shifted within Transformer blocks. This mechanism makes Swin Transformers more suitable for processing high-resolution images. Swin Transformers are found effective in image classification, object detection, and semantic segmentation problems.

* Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030. https://arxiv.org/abs/2103.14030.

# Contact

Yingkai (Kyle) Sha <<yingkai@eoas.ubc.ca>> <<yingkaisha@gmail.com>>

The work is benefited from:
* The official Pytorch implementation of Swin-Transformers [[link](https://github.com/microsoft/Swin-Transformer)].
* Swin-Transformer-TF [[link](https://github.com/rishigami/Swin-Transformer-TF)].


# License

[MIT License](https://github.com/yingkaisha/swin_transformer_keras/blob/main/LICENSE)
