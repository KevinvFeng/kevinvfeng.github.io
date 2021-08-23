---
title: torchdrug 使用指南 采坑记录(1)
date: 2021-08-19 20:47:55
author: Kevin_Feng
tags:
 - torchdrug
 - Machine Learning
---
本系列介绍了 Torchdrug(目前 V0.1) 的基本用法
<!--more-->
## 安装
目前(2021年08月19日) 使用 conda 安装在某些情况会报错(主要原因是 conda 安装 pytorch-scatter 报错), 建议直接参照 pip 的安装方式. 
1. 构建 conda 环境
```conda create --name=torchdrug python=3.8```
2. 安装 pytorch
参照 pytorch 官网安装对应的 gpu 版本即可
3. 根据 pytorch 的版本安装对应的pytorch-geometric [点我查看版本](https://pytorch-geometric.com/whl/)
```shell
# PyTorch 1.9.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html# PyTorch 1.8.0/1.8.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
```
4. 安装 torchdrug
```shell
git clone https://github.com/DeepGraphLearning/torchdrug
cd torchdrug
python setup.py install
```
5. 安装 pypi 版 rdkit
```shell
pip install rdkit-pypi
```
6. 检查是否安装完毕
```python
import torchdrug as td
```

## Get Started
下面将以一个分子性质预测的任务作为例子初步探索 torchdrug 的基本使用方法
### 导入包
```python
import torch
import torchdrug as td
from torchdrug import data, datasets
from torchdrug import core, models, tasks
from torchdrug import utils
from torch.nn import functional as F
import matplotlib.pyplot as plt
%matplotlib notebook #if using jupyter notebook
```
### 构建数据集并展示
```python
dataset = datasets.ClinTox("/path_to_dataset/clintox/")
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
```
```python
graphs = []
labels = []
for i in range(4): # 展示前四个分子
    sample = dataset[i]
    graphs.append(sample.pop("graph"))
    label = ["%s: %d" % (k, v) for k, v in sample.items()]
    label = ", ".join(label)
    labels.append(label)
graph = data.Molecule.pack(graphs)
graph.visualize(labels, num_row=1)
```
### 构建模型以及 task
```python
model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))
```
### 训练
```python
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=1024)
solver.train(num_epoch=100)
solver.evaluate("valid")
```
### 验证训练好的模型
```python
from torchdrug import utils
from torch.nn import functional as F
samples = []
categories = set()
# 这里官方文档上写的代码有点问题
# Molecule pop('graph') 之后就没有这个属性了 在后面会报错
samples = data.graph_collate(valid_set[:2]) # 取前两个分子作为展示
samples = utils.cuda(samples)
preds = F.sigmoid(task.predict(samples))
targets = task.target(samples)

titles = []
for pred, target in zip(preds, targets):
    pred = ", ".join(["%.2f" % p for p in pred])
    target = ", ".join(["%d" % t for t in target])
    titles.append("predict: %s\ntarget: %s" % (pred, target))
graph = samples["graph"]
graph.visualize(titles, figure_size=(3, 3.5), num_row=1)
# 个人感觉这个分子画出来的效果有点乱也有点丑.
```
## 一些常见问题
- visualize() 没有显示
在 Matplotlib 下可以添加一行```%matplotlib notebook```
之后再次运行即可
- 官方文档的 Get Started 报错
可以参照本文的代码进行尝试, 本文对官方文档中代码的一些问题进行的修改, 亲测可以得出正常的结果.