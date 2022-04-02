# NNDL_hw1
A 2-layer fully-connected neural network on MNIST

## PART 0 数据读取
数据的读取参考了https://zhuanlan.zhihu.com/p/163254171 中读取MNIST数据的方法，对测试集样本的读取方法稍作改动。在NNLP_hw1_all_codes.py文件中data_fetch_preprocessing()实现数据的读取；

## PART 1 调参
根据预训练结果，选择对起始学习率\alpha在[0.05, 0.1, 0.2]范围内，对隐藏层节点个数p在[100, 200, 300]范围内以及对L2正则化系数在[0, 0.005, 0.01]范围内进行调参。根据模型在测试集表现，最终选择起始学习率0.05，隐藏层节点数200，L2正则项系数0的参数组合。这部分工作由NNLP_hw1_all_codes.py中modelSelection()函数实现；
