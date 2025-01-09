# 第10周工作记录

---
## 2025-01-06
计划：针对01-05日发现的问题进行修改。该部分问题详述见TODO（暂时）和[第9周工作记录](week9ResearchRecord.MD)。

## 2025-01-08
在coco_validation.py的基础上修改，在SARDet-100K数据集上实现模型验证[SARDet_validation.py](../SARDet_validation.py)。  
验证结果见[第一次训练模型验证结果](../runs/val/RetinaNet模型验证结果.txt)

了解到，MMDetection集成了常用模型的代码，包括但不限于RetinaNet，因此考虑接下来继续修改此代码，还是利用MMDetection重新实现。
