# class-activate-map
a tool can visual why do CNN-based neural network work like this?

# 用途
1. 原因分析。对于文本分类任务，当神经网络（具有CNN结构）输出类别时，获得输入文本中的各个片段（如字/词）对于作出该分类的影响程度，同时可视化。进一步可以查看某些类别是否对特定文本片段过于敏感导致过拟合，指导调参；     
2. 模式提取。文本分类场景下，对于预测结果均为类别c的文本集合，将这些文本中对类别c影响程度大的片段作为关键模式提取出来，可供其他任务使用。  
# 样例
* 例子为文本多标签多分类任务（多标签即一条文本可以属于多个类别），类别包括：健康、情感、教育、数码、商业、旅游等40个类别。  
* 模型结构为：`char-embedding + biLSTM + self-attention + CNN + multi-sigmod-output`. 

1. 样例1.  
`输入文本：现在那个品牌的手机比较好用.`.  
`top3输出类别：数码（sigmod=0.99），购物（0.58），商业（0.01）.`.  
可视化结果（图中百分比及颜色深浅表示对应的字对于类别的重要性） 
>输出类别为`数码`,可见影响程度最大的文本片段是`手机`.   
![](https://github.com/sciencefren/class-activate-map/blob/master/example_imgs/example2_数码.png)  

>输出类别为`商业`，影响最大的文本片段是`品牌`和`手机`.  
![](https://github.com/sciencefren/class-activate-map/blob/master/example_imgs/example2_商业.png)  

2. 样例2.  
`输入文本：我想上深圳大学，不知道那里天气怎么样.`.  
`top3输出类别：教育（sigmod=0.99），就业（0.02），旅游（0.01）.`.  
>输出类别为`教育`,影响程度最大的文本片段是`大学`.   
![](https://github.com/sciencefren/class-activate-map/blob/master/example_imgs/example1_教育.png)  

>输出类别为`旅游`，影响最大的文本片段是`深圳大学`和`天气`.  
![](https://github.com/sciencefren/class-activate-map/blob/master/example_imgs/example1_旅游.png)  

# 使用说明
* `src/ClassActivateMap.py`包含获取文本片段重要性的类及方法。   
```
class ClassActivateMap(ckpt_fp, label2id_dct, input_tensor_name_lst, conv_layer_tensor_name2filter_size_dct, logits_tensor_name)  
>ckpt_fp: 训练保存的checkpoint文件夹，如checkpoint_textclassify_01  
>label2id_dct: 类别标签名称到id的映射(dict格式），如{'教育': 0, '科技': 1, ...}  
>input_tensor_name_lst： 模型输入的tensor名称(list格式)，通常为图结构中定义的placeholder名称，  
                        如['input_text:0', 'input_sentence_len:0', 'dropout_keep_prob:0']。  
>conv_layer_tensor_name2filter_size_dct: CNN层filter名称到对应filter map大小的映射(dict格式)，  
                                        如{'conv-maxpool-1/relu:0': 1, 'conv-maxpool-3/relu:0':3, ...}  
>logits_tensor_name: 模型分类层（输出层）的tensor名称(string格式)，如'output/logit:0'  

ClassActivateMap.
ClassActivateMap.get_text_final_cam(self, input_value_lst, target_label, max_sentence_len, true_length)
```
* 
