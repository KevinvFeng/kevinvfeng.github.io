---
title: MG-Bert (Molecular Graph BERT)
date: 2021-07-18 14:08:48
author: Kevin Feng
tags:
- Machine Learning
- QSAR
---

## TL;DR
作者将Bert的思想与GNN结合起来，采用了预训练的方式提升了分子属性预测的准确率. 同时作者的实验结果表明,氢在分子图中同样很重要.
<!--more-->

## 模型结构及训练方法等

### 输入
1. 分子图, 共有13个原子 外加UNK, Global作者额外引入的超级节点, 预训练用的MASK. 共16个token

### 模型架构
![](model.png)
整体同bert一样,有三个模块,embedding / Transformer encoder / task-related output layer.

#### Embedding
MG-Bert取消了原生Bert中的Position Embedding, 理由是对于分子而言,我们应该更多的聚焦在通过化学键相连的原子间位置关系,而不是NLP中的Global Attention.

#### Transformer Encoder
由于在embedding中取消了position embeddding, 所以需要在这部分把分子的结构信息补回来. 补回来的方法就是通过引入local message passing mechanism.
作者称他们通过包含了res-connection的bert解决了 oversmoothing的问题.
在这部分作者还引入了SuperNode.一个连接着所有原子的节点(额外引入的), 会被MPN传播.这个节点有两重意义,一是帮助分子进行远距离的消息传播, 二是我们可以用这个节点表示整个分子,在训练下游任务的时候可以从这里获取信息. 

#### task-related output layer
这部分对于不同任务 包括预训练还是finetune都是不同的, 所以预训练只是训练了embedding以及transformer encoder部分.
对于finetune任务就是在Global(SuperNode)后面接了两层dense layer.

### 预训练

#### 方法
首先,作者提出了类似bert的预训练的方法, 不过只做了MLM(masked language model)部分, 通过RDKit把分子转化成2d无向图, 给每个原子连接一个supernode(也就是上文中提到的[Global]). 随机将分子中15%的原子(至少1个)替换掉: 80%概率换成[MASK], 10%概率换成别的,以及10%的概率不变.这部分跟原生bert一致. 目标是去预测这个被修改的原子本身是什么.
__细节:__ batch SGD / Adam / lr=1e-4 / batch_size=256 / epoch=10

#### 训练集
从ChEMBL数据集(1.7million个分子)中随机选择了90%作为训练集, 其余10%作为验证集, 训练10个epoch即可. 注: 用moses,感觉同样能达成效果, 可以尝试使用更大的数据集进行预训练.

## 训练及评估
1. 与训练之后,  预训练用的head就会被一处, 然后添加两层dense layer 用于其他任务.
__细节:__ 使用dropout避免过拟合,作者认为这个dropout非常重要. dropout rate需要设为[0-0.5]. 用了Adam, batch_size以及lr分别从一下的选项中进行选择{16,32,64},{1e-5,5e-5,1e-4}, 具体用那种根据不同的任务进行选择.
2. 评价: 对于回归模型采用R<sup>2</sup> 而分类问题则使用了ROC-AUC. epoch最大设置为100,并采用了early stop.

## 最终效果
11个分子性质预测的SOTA.可见预训练还是很重要的.

## 一些结论
- 预训练比不预训练好很多 R<sup>2</sup> 或AUC相差2%以上
- 作者认为将分子图中没有引入的氢补全会比较重要,因为它可以用于在预训练阶段判断 六个碳相连的是苯环还是普通环, 对于回归任务来说也有很大影响. (不过我觉得如果把共价键的信息补全可以达到相同的效果,不需要补氢)
- 经过预训练后, 模型可以很好地编码不同原子类型
![](tsne.png)
如上图所示, 作者随机抽取了1000个预训练集中的分子(差不多22000个原子),经过transformer encoder后的信息. 经过t-SNE可视化之后就如上图所示. 可以很简单的看出即使是相同的原子, 在分子图中表示的结果也是不太一样的. 单一原子会被分成很多不同的cluster. 另外锁着还随机抽取了一个分子进行更深入的研究.结果表明,这个模型可以很明显的识别苯环中的碳以及非苯环中的碳. 这表明了模型可以较为容易的识别一阶邻域信息(跟这个原子相邻的原子的信息)甚至更高级的信息.
- 通过attention, 我们可以可视化模型在进行性质预测的时候跟关注分子层面的哪些部分. 如图所示
![](att.png)
