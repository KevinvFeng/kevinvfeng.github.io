---
title: HJT-VAE模型
date: 2021-05-25 16:44:56
author: Kevin Feng
tags:
- Machine Learning
---
以往的分子图模型都是围绕着粒度较小的单元进行拆分，比如一个原子或者一个小环。而HJT-VAE模型，提供了多粒度的分子表示，从原子层面到connected motifs(连接模态?)
<!--more-->
## JT-VAE
说 HJT-VAE 得先说 JT-VAE 这是同一个作者之前的研究.
### 其他模型现状
#### Smiles
以往大多数基于 SMILES 的模型会有一个先天的问题,同一个分子有多个 SMILES 表示,导致每个字符之间的位置关系并不完美.同理,两个相似的分子的 SMILES 的距离(edit distance)可能相差很大.这会导致 VAE 类模型并不能很好的学习到每个字符的平滑 embedding. 
#### 原子粒度
其他模型如果采用原子层面的图模型表示的话,也会有一个问题:生成一个苯环结构会先依次生成 6 个原子而不是直接放一个苯环,这几步中间过程生成出的分子很可能是无效的.
#### 作者解决的问题
所以作者就提出了基于图片,并且基于不同粒度的 VAE 模型:JT-VAE.
### 解决方案
#### 前提条件
- 训练集足够大,保证片段能够覆盖需求
## 其他模型的问题
传统分子图模型，在小分子上的表现还是可以的，但是因为梯度的问题，在大分子上，比如polymer(聚合物)就表现不行了. 对于聚合物而言,它有着非常清晰的层次结构.如下图左侧所示,这图可以分成了几部分,中间这一块称之为 Structural motifs.
![](HJT-VAE/hjt-vae-1.png)
## HJT-VAE 解决办法
作者提出了基于 motif 的hierarchical-VAE. 这个 motif 是预先就提取好的.在进行分子生成的时候,分子一点一点的往上添加 motif,从大到小.在 decoder 每次添加新 motif的时候要考虑三件事情:
1. new motif selection, which part of it attaches, and the points of contact with
the current molecule
### Motif Extraction
作者把每一个 Motif 看成一个分子图中的子图,所有的 Motif 联合起来会覆盖整个分子.在提取 Motif 的时候会执行以下步骤:
1. 断开所有的桥键(断开这个键并不会改变这个分子的有效性),此时会把分子图拆开成几个不连通的子图
2. 提取出训练集中出现次数大于等于 100 的子图
3. 如果某个字图没有被选为 Motif 则把他分解成环和键并作为 motif.
4. 将所有 motif 收集记为 Vs
### 层次图生成
TODO:
## 每层详解
### Motif 层
这一层记录了 Motif 的信息以及 Motif 之前的大致连接.
注意这里是树结构存储的,理由是 encoder 中就是这样
### Attachment 层
这一层决定了 Motif 中的什么节点(原子)与其他 Motif 连接.
### Atom 层
原子层记录了原子之间是如何连接的,记录了$(u,v)$其中 $u,v$ 是原子.同时还记录了原子之间的键 $b_{uv}$
### 层间关系
每一层之间其实都是个图,作者还在层与层之间做了连接,比如从原子到 Attachment 层,以及 Attachment 到 Motif 层,这些层间的连接同样会通过 MPN 传播. 对于 Attachment 和 Motif 其编码方式都是 One-hot

> Hierarchical Generation of Molecular Graphs using Structural Motifs