# RetinaNet算法在SAR目标检测数据集上的实验

--- 

## 训练
在文件[train.py](train.py)中配置训练参数并运行。

## 评估
将要评估的模型存储于文件夹[weights](./weights)中后，
在文件[val_batch.py](val_batch.py)中的main函数开头，
设置要评估的模型的文件名和骨干网深度（一一对应），
然后运行，即可自动评估多个模型。

## 获取模型参数量、运算量等的总结
在文件[modelSummary.py](modelSummary.py)中的主程序入口处配置模型路径和输入图像大小，
运行后即可在终端获得模型结构简报、参数量、MACs、GFLOPs、FPS等信息。  
**说明**：
1. 所有计算均在`test_model()`函数中进行，如对结果有疑问，请检查该函数。
2. FLOPs由式`FLOPs = MACs / 2`计算得到。
3. FPS和每张图像的推理时长由100张图像依次输入模型后平均得到，计算方法为
    ```
        total_time = end_time - start_time  # 总耗时
        fps = num_images / total_time       # FPS
        inf_time = total_time / num_images  # 每张图像的推理时长
    ```
4. 模型结构的总结，可以在`test_model()`函数中调取`summary()`函数处配置，相关参数含义可参阅[此处](https://blog.csdn.net/qq_40206371/article/details/120558250)。




