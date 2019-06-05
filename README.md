# DeepSORT_Features

多目标跟踪论文 Deep SORT 特征训练PyTorch实现
主要是用于训练特征的部分，使用PyTorch实现

Market-1501数据先要用prepare.py处理一下，符合PyTorch加载方式
也就是产生一些子文件夹，让训练文件夹和测试文件夹分开
MARS数据集不用动

CNNArchitecture.py 描述了整个模型的结构

FunctionsTest.py主要用于代码中一些函数的测试
train.py 用于训练
