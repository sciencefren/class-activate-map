# class-activate-map
a tool can visual why do CNN-based neural network work like this?

# 用途
1. 原因分析。对于文本分类任务，当神经网络（具有CNN结构）输出类别时，获得输入文本中的各个片段（如字/词）对于作出该分类的影响程度，同时可视化；  
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

>输出类别为`旅游`，影响最大的文本片段是`天气`.  
![](https://github.com/sciencefren/class-activate-map/blob/master/example_imgs/example1_旅游.png)  
