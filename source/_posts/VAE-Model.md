---
title: VAE模型及其变种
date: 2021-05-21 11:22:19
author: Kevin Feng
tags:
- Machine Learing
---
# 背景知识
## AutoEncoder
AutoEncoder是一个很基本的网络结构，输入是X输出希望也是X，不过中间会有一层latent hidden layer的维度要远小于输入和输出维度。一般这个中间层的维度也就几十左右。
### AutoEncoder的作用
AE的用途有不少，主要的核心任务是可以用于降维，类似PCA。 同时也可以生成不同的结果用于数据增强。
# VAE模型
## VAE基础结构
VAE在AE的模型上稍微改变了对latent hidden layer。将latent z 拆成了两个，mu以及logvar，之后再将它们合起来。

![](vae_sample_with_kld.png)
图为用vae训练后根据不同mu，sample出的图片。可以明显的看出图片渐变的过程。我们希望渐变的过程是平滑的，这样就可以方便的找到两点之间的结果用于产生新的图片
![](vae_sample_no_kld.png)
我发现不用kld其实也差不多，但是下面的两张图区别就比较大了
![](vae_input_scatter_with_kld.png)
![](vae_input_scatter_no_kld.png)
不加kld的话，z的数值会比较大， 从而导致进行合理sample的时候会更困难，不利于sample.