# RetrievalBert

## 模型训练及测试：
```
python main.py
```

## 一些重要的文件：
- main.py 首先预训练RoBERTa生成Datastore（fine_tune_pretrain_model_generate_datastore函数），然后结合已生成的Datastore继续训练（fine_tune_with_knn函数），后者可以通过指定fixed_finetune、fixed_knn和only_knn参数来确定是否保持预训练模型参数不变、是否保持datastore不变以及是否只使用datastore预测结果。
- datasets.py 数据预处理&Dataset定义
- models.py 模型定义
- knn.py Datastore定义
- parameters.py 定义超参、文件位置等
