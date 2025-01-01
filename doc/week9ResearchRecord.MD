# 第9周工作记录

## 2025.01.01
修改代码以适应SARDet-100K数据集。  
成功部署RetinaNet算法，开始在SARDet-100K数据集上训练。  
训练过程输出如下图。  
![RetinaNet第一次开始训练.png](figures/RetinaNet%E7%AC%AC%E4%B8%80%E6%AC%A1%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83.png)

**仍然存在的问题：**  
由于原程序根据json文件读取图片，因此无法使用削减的训练集训练模型。  
**解决方案：** 编写python程序，根据实际数据集重新构建json文件供训练使用。  

