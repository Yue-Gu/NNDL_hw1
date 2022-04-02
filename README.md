# NNDL_hw1
A 2-layer fully-connected neural network on MNIST

文件中调参、模型训练部分的正常运行需要下载http://yann.lecun.com/exdb/mnist/ 中的4个压缩包，并将解压缩后的文件与代码放在同一目录下。调参需运行PART0与PART1，模型训练需运行PART0与PART2。文件中模型导入与测试、可视化部分需要下载https://pan.baidu.com/s/1XaLjJj9LrcDiZ9xw9rF4Eg （提取码：6666）中的4个txt文件。模型导入与测试集准确率需要运行PART4，可视化需要运行PART5（loss曲线、accuracy曲线等）以及PART6（网络中权重矩阵的可视化）。下面详细介绍各个板块的主要函数：

## PART 0 数据读取
数据的读取参考了https://zhuanlan.zhihu.com/p/163254171 中读取MNIST数据的方法，对测试集样本的读取方法稍作改动。在NNLP_hw1_all_codes.py文件中data_fetch_preprocessing()实现数据的读取；

## PART 1 调参
根据预训练结果，选择对起始学习率\alpha在[0.05, 0.1, 0.2]范围内，对隐藏层节点个数p在[100, 200, 300]范围内以及对L2正则化系数在[0, 0.005, 0.01]范围内进行调参。根据模型在测试集表现，最终选择起始学习率0.05，隐藏层节点数200，L2正则项系数0的参数组合。这部分工作由NNLP_hw1_all_codes.py中modelSelection()函数实现；
