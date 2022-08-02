## 微博爬虫与评论情感分析

### 1.使用说明

### 1.1根据关键词搜索爬取大量微博数据

在`search_spider`文件夹下，运行`search_start.py`文件，需要提前获取登录[微博搜索](https://s.weibo.com)的cookie，以及手动输入关键词。

### 1.2根据某一话题下的微博数据爬取微博评论

在`comment_spider`文件夹下，运行`comment_start.py`文件，需要提前获取登录[微博](https://weibo.cn)的cookie，以及手动输入关键词。

### 1.3对微博评论进行情感分析

在`emotion_analysis`文件夹下，运行`bert_bilstm.py`文件，修改关键词`predict_main('全球性别不平等报告')`可以直接进行预测，训练模型则需要将注释取消：

```python
file = './data/usual_train.txt'
train_main(model_config, file)
```

## 2.方法简介

根据微博热搜词条爬取相关微博下的评论文本数据，将评论送入基于BERT训练的情感倾向分类模型。

BERT模型是一个多层双向的Transformer编码器，实现方式主要分为预训练和微调两个步骤。BERT模型参数先用预训练参数进行初始化，再利用“文本-情感标注”数据进行模型微调。 本项目实现情感的细粒度分类。

## 3.数据集以及参考代码

数据集来源：[SMP2020](https://smp2020ewect.github.io/)

本次实验的训练数据存放在`emotion_analysis/data/usual_train.txt`中。

代码参考：[Weibo_Spider](https://github.com/WDevin/Weibo_Spider)
