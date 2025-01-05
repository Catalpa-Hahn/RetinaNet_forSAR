# RetinaNet_forSAR版本管理记录

## 2025-01-01更新
已能在SARDet_100K数据集上开展训练，效果未知。  
由于原程序根据json文件读取图片，因此无法使用削减的训练集训练模型。

## 2025-01-03更新
已能在削减的SARDet-100K数据集上开展训练，效果待验证。

## 2025-01-05更新
发现一些存在的问题，主要为：  
1、loss的存储和直观表现（曲线图）；  
2、缺少针对SARDet-100K数据集的模型评估过程，需添加；  
3、缺少判断最优模型的过程，需添加。  
上述问题已写入TODO。

## 其他说明
修改、调试过程详细记录，以及一些详细说明见：  
(1) [第9周工作记录](doc/week9ResearchRecord.MD)

## 参考资料
(1) 论文和代码详解：https://zhuanlan.zhihu.com/p/143877125  
(2) 原工程地址：https://link.zhihu.com/?target=https%3A//github.com/yhenon/pytorch-retinanet