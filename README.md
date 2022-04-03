# NNDL_hw1
A 2-layer fully-connected neural network on MNIST

文件中调参、模型训练部分的正常运行需要下载http://yann.lecun.com/exdb/mnist/ 中的4个压缩包，并将解压缩后的文件与代码放在同一目录下。调参需运行PART0与PART1，模型训练需运行PART0与PART2。文件中模型导入与测试、可视化部分需要下载https://pan.baidu.com/s/1XaLjJj9LrcDiZ9xw9rF4Eg （提取码：6666）中的4个txt文件。模型导入与测试集准确率需要运行PART4，可视化需要运行PART5（loss曲线、accuracy曲线等）以及PART6（网络中权重矩阵的可视化）。下面详细介绍各个板块的主要函数：

## PART 0 数据读取
数据的读取参考了https://zhuanlan.zhihu.com/p/163254171 中读取MNIST数据的方法，对测试集样本的读取方法稍作改动。在NNLP_hw1_all_codes.py文件中data_fetch_preprocessing()实现数据的读取。

## PART 1 调参
根据预训练结果，选择对起始学习率\alpha在[0.05, 0.1, 0.2]范围内，对隐藏层节点个数p在[100, 200, 300]范围内以及对L2正则化系数在[0, 0.005, 0.01]范围内进行调参。根据模型在测试集表现，最终选择起始学习率0.05，隐藏层节点数200，L2正则项系数0的参数组合。特别的，虽然这里正则项系数通过调参选择为0，但是损失函数计算、反向传播中梯度计算都考虑了正则项，计算式L2正则项系数参数代入0即可。

这部分工作由NNLP_hw1_all_codes.py中modelSelection()函数实现。

## PART 2 模型训练
首先定义NeuralNetwork()类，该类中定义了损失函数（均方误差损失函数+L2正则项），激活函数（softmax），前馈计算与各个参数的反向传播函数。首先用输入层、隐藏层、输出层、学习率、epoch以及正则项系数初始化NeuralNetwork类。

随后，使用该类的trainNN()函数，训练模型。该函数用到了该类中的前馈计算、反向传播等函数。最终输出结果为一个向量，[W1, W2, b1, b2, train_loss, test_loss, test_acc]。其中W1, W2, b1, b2为最后一个epoch结束后网络中的参数；train_loss为存储每个epoch中训练样本loss的向量，长度为epoch数目；test_loss为存储每轮epoch后模型在测试样本中loss的向量，长度为epoch数目；test_acc为存储每轮epoch后模型在测试样本中准确率的向量，长度为epoch数。

最终trainNN()输出结果保存到output中。

## PART 3 模型存储
将PART 2中的结果分别保存到文本文件中。

## PART 4 导入模型并测试
使用numpy库中的loadtxt函数导入W1, W2, b1, b2中，存为array。之后使用testModel()函数计算模型在测试集上的准确率。testModel()中，将输入样本使用softmax激活函数以及前馈计算，得到最终的预测结果。之后计算并返回测试样本中的准确率。

## PART 5 loss曲线与accuracy曲线
导入PART 3中存储的test_loss，train_loss与test_acc向量，绘制折线图。

## PART 6 网络权重可视化
