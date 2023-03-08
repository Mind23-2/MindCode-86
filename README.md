# 目录

- [目录](#目录)
- [STGCN 介绍](#STGCN-介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本介绍](#脚本介绍)
    - [脚本以及简单代码](#脚本以及简单代码)
    - [脚本参数](#脚本参数)
    - [训练步骤](#训练步骤)
        - [训练](#训练)
    - [评估步骤](#评估步骤)
        - [评估](#评估)
- [模型介绍](#模型介绍)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [Inference Performance](#inference-performance)
- [随机事件介绍](#随机事件介绍)
- [ModelZoo 主页](#ModelZoo-主页)

# STGCN 介绍

STGCN主要用于交通预测领域，是一种时空卷积网络。在STGCN文章中提出一种新颖的深度学习框架——时空图卷积网络（STGCN），解决在通领域的时间序列预测问题。在定义图上的问题，并用纯卷积结构建立模型，这使得使用更少的参数能带来更快的训练速度。STGCN通过建模多尺度交通网络有效捕获全面的时空相关性，且在各种真实世界交通数据集始终保持SOTA。

[Paper](https://arxiv.org/abs/1709.04875): Bing yu, Haoteng Yin, and Zhanxing Zhu. "Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting." Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2017.

# 模型架构

STGCN模型结构是由两个时空卷积快和一个输出层构成。如上图所示，左侧是STGCN网络模型框架，中间为时空卷积块，右侧为时域卷积块。空域卷积块有两种不同卷积方式，分别为：Cheb和GCN。

# 数据集

Dataset used:

PeMED7(PeMSD7-m、PeMSD7-L)
BJER4

由于数据集下载原因，只找到了[PeMSD7-M](https://github.com/hazdzz/STGCN/tree/main/data/train/road_traffic/pemsd7-m)数据集。

# 环境要求

- 硬件（Ascend/GPU）
    - 需要准备具有Ascend或GPU处理能力的硬件环境. 如需使用Ascend，可以发送 [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) 到ascend@huawei.com。一旦批准，你就可以使用此资源
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需获取更多信息，请查看如下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速开始

在通过官方网站安装MindSpore之后，你可以通过如下步骤开始训练以及评估：

- running on Ascend with default parameters

  ```python
  # 单卡训练
  python train.py --train_url="" --data_url="" --run_distribute=False --run_modelarts=False --graph_conv_type="chebgcn" --n_pred=9

  # 多卡训练
  bash scripts/run_distribute_train.sh train_code_path data_path n_pred graph_conv_type
  ```

# 脚本介绍

## 脚本以及简单代码

```python
├── STGCN
    ├── scripts
        ├── run_distribute_train.sh     //traing on Ascend with 8P
        ├── run_eval_ascend.sh     //testing on Ascend
    ├── src
        ├── model
            ├──layers.py       // model layer
            ├──metric.py          // network with losscell
            ├──models.py          // network model
        ├──config.py       // parameter
        ├──dataloder.py          // creating dataset
        ├──utility.py          // calculate laplacian matrix and evaluate metric
    ├── train.py                // traing network
    ├── test.py                 // tesing network performance
    ├── README.md                 // descriptions
```

## 脚本参数

训练以及评估的参数可以在config.py中设置

- config for STGCN

  ```python
     stgcn_chebconv_45min_cfg = edict({
    'learning_rate': 0.003,
    'n_his': 12,
    'n_pred': 9,
    'n_vertex': 228,
    'epochs': 500,
    'batch_size': 8,
    'decay_epoch': 10,
    'gamma': 0.7,
    'stblock_num': 2,
    'Ks': 2,
    'Kt': 3,
    'time_intvl': 5,
    'drop_rate': 0.5,
    'weight_decay_rate': 0.0005,
    'gated_act_func':"glu",
    'graph_conv_type': "chebconv",
    'mat_type': "wid_sym_normd_lap_mat",
    })
  ```

如需查看更多信息，请查看`config.py`.

## 训练步骤

### 训练

- running on Ascend

  ```python
  #1P训练
  python train.py --train_url="" --data_url="" --run_distribute=False --run_modelarts=True --graph_conv_type="chebgcn" --n_pred=9
  #8P训练
  bash scripts/run_distribute_train.sh train_code_path data_path n_pred graph_conv_type
  ```

  8P训练时需要将RANK_TABLE_FILE放在scripts文件夹中，RANK_TABLE_FILE[生成方法](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)

  训练时，训练过程中的epch和step以及此时的loss和精确度会呈现在终端上：

  ```python
  epoch: 1 step: 139, loss is 0.429
  epoch time: 203885.163 ms, per step time: 1466.800 ms
  epoch: 2 step: 139, loss is 0.2097
  epoch time: 6330.939 ms, per step time: 45.546 ms
  epoch: 3 step: 139, loss is 0.4192
  epoch time: 6364.882 ms, per step time: 45.791 ms
  epoch: 4 step: 139, loss is 0.2917
  epoch time: 6378.299 ms, per step time: 45.887 ms
  epoch: 5 step: 139, loss is 0.2365
  epoch time: 6369.215 ms, per step time: 45.822 ms
  epoch: 6 step: 139, loss is 0.2269
  epoch time: 6389.238 ms, per step time: 45.966 ms
  epoch: 7 step: 139, loss is 0.3071
  epoch time: 6365.901 ms, per step time: 45.798 ms
  epoch: 8 step: 139, loss is 0.2336
  epoch time: 6358.127 ms, per step time: 45.742 ms
  epoch: 9 step: 139, loss is 0.2812
  epoch time: 6333.794 ms, per step time: 45.567 ms
  epoch: 10 step: 139, loss is 0.2622
  epoch time: 6334.013 ms, per step time: 45.568 ms
  ...
  ```

  此模型的checkpoint存储在train_url路径中

## 评估步骤

### 评估

- 在Ascend上使用PeMSD7-m 测试集进行评估

  在使用命令运行时，需要传入模型参数地址、模型参数名称、空域卷积方式、预测时段。

  ```python
  python test.py --run_modelarts=False --run_distribute=False --device_id=0 --ckpt_url="" --ckpt_name="" --graph_conv_type="" --n_pred=9
  #使用脚本评估
  bash scripts/run_eval_ascend.sh data_path ckpt_url ckpt_name device_id graph_conv_type n_pred
  ```

  以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以如下方式呈现：

  ```python
  MAE 3.23 | MAPE 8.32 | RMSE 6.06
  ```

# 模型介绍

## 性能

### 评估性能

#### STGCN on PeMSD7-m (Cheb,n_pred=9)

| Parameters                 | ModelArts
| -------------------------- | -----------------------------------------------------------
| Model Version              | STGCN
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G
| uploaded Date              | 05/07/2021 (month/day/year)
| MindSpore Version          | 1.2.0
| Dataset                    | PeMSD7-m
| Training Parameters        | epoch=500, steps=139, batch_size = 8, lr=0.003
| Optimizer                  | AdamWeightDecay
| Loss Function              | MES Loss
| outputs                    | probability
| Loss                       | 0.183
| Speed                      | 8pc: 45.601 ms/step;
| Scripts                    | [STGCN script]

### Inference Performance

#### STGCN on PeMSD7-m (Cheb,n_pred=9)

| Parameters          | Ascend
| ------------------- | ---------------------------
| Model Version       | STGCN
| Resource            | Ascend 910
| Uploaded Date       | 05/07/2021 (month/day/year)
| MindSpore Version   | 1.2.0
| Dataset             | PeMSD7-m
| batch_size          | 8
| outputs             | probability
| MAE                 | 3.23
| MAPE                | 8.32
| RMSE                | 6.06
| Model for inference | about 6M(.ckpt fil)

# 随机事件介绍

我们在train.py中设置了随机种子

# ModelZoo 主页

 请查看官方网站 [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).