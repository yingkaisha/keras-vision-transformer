# keras-vision-transformer

This repository contains the `tensorflow.keras` implementation of the Swin Transformer (Liu et al., 2021) and its applications to benchmark datasets.

* Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030. https://arxiv.org/abs/2103.14030.

* Hu, C., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.

# Notebooks

Note: the Swin-UNET implementation is experimental

* MNIST image classification with Swin Transformers [[link](https://github.com/yingkaisha/keras-vision-transformer/blob/main/examples/Swin_Transformer_MNIST.ipynb)]
* Oxford IIIT Pet image Segmentation with Swin-UNET [[link](https://github.com/yingkaisha/keras-vision-transformer/blob/main/examples/Swin_UNET_oxford_iiit.ipynb)]

# Dependencies

* TensorFlow 2.5.0, Keras 2.5.0, Numpy 1.19.5.

# Overview

Swin Transformers are Transformer-based computer vision models that feature self-attention with shift-windows. Compared to other vision transformer variants, which compute embedded patches (tokens) globally, the Swin Transformer computes token subsets through non-overlapping windows that are alternatively shifted within Transformer blocks. This mechanism makes Swin Transformers more suitable for processing high-resolution images. Swin Transformers have shown effectiveness in image classification, object detection, and semantic segmentation problems.

# Contact

Yingkai (Kyle) Sha <<yingkai@eoas.ubc.ca>> <<yingkaisha@gmail.com>>

The work is benefited from:
* The official Pytorch implementation of Swin-Transformers [[link](https://github.com/microsoft/Swin-Transformer)].
* Swin-Transformer-TF [[link](https://github.com/rishigami/Swin-Transformer-TF)].

# License

[MIT License](https://github.com/yingkaisha/swin_transformer_keras/blob/main/LICENSE)
