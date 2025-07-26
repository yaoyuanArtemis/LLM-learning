import torch     # 2.7.1
import torch.nn as nn
import torch.nn.functional as F
import os
import requests  # 2.32.4



## 请求数据集
if not os.path.exists("sales-textbooks.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("sales-textbooks.txt","wb") as f:
        f.write(requests.get(url).content)

with open("sales-textbooks.txt","r") as f:
    text = f.read()
    # print("🚀 text:",text)


## tokenize化
import tiktoken
encoding = tiktoken.get_encoding("o200k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text,dtype=torch.long)


## split train_set and validation_set
train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validate_data = tokenized_text[train_index:]

## hyperparameter
context_length = 16 
d_modal = 64
batch_size = 4
num_heads = 4

## 训练数据中整批次整理好数据
data = train_data
idxs = torch.randint(low=0,high=len(data) - context_length,size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])
y_batch = torch.stack([data[idx+1:idx + context_length+1] for idx in idxs])

import pandas as pd
# print(pd.DataFrame(x_batch[0].numpy()))

## create Input Embedding Table
max_token_value = tokenized_text.max().item()
input_embedding_lookup_table = nn.Embedding(max_token_value+1,d_modal)
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch) # y_batch_embedding.shape = (4 * 16 * 64)

## positional encoding
import math
positinal_encoding_lookup_table = torch.zeros(context_length,d_modal)
position = torch.arange(0,context_length,dtype=torch.float).unsqueeze(1)
div_item = torch.exp(torch.arange(0,d_modal,2).float() * (-math.log(10000.0) / d_modal))

positinal_encoding_lookup_table[:,0::2] = torch.sin(position * div_item)
positinal_encoding_lookup_table[:,1::2] = torch.cos(position * div_item)
positinal_encoding_lookup_table = positinal_encoding_lookup_table.unsqueeze(0).expand(batch_size,-1,-1)

x = x_batch_embedding + positinal_encoding_lookup_table
y = y_batch_embedding + positinal_encoding_lookup_table
# print(pd.DataFrame(x[0].detach().numpy()))



# head-multi attetion
## get Q、K、V
Wq = nn.Linear(d_modal,d_modal)
Wk = nn.Linear(d_modal,d_modal)
Wv = nn.Linear(d_modal,d_modal)


Q = Wq(x)
K = Wk(x)
V = Wv(x)


## 切分Q、K、V
Q = Q.view(batch_size,context_length,num_heads,d_modal//num_heads).permute(0,2,1,3)
K = K.view(batch_size,context_length,num_heads,d_modal//num_heads).permute(0,2,1,3)
V = V.view(batch_size,context_length,num_heads,d_modal//num_heads).permute(0,2,1,3)

## Calculate QK^T
output = Q @ K.transpose(-2,-1) / math.sqrt(d_modal//num_heads)

## 注意力机制中添加mask
mask = torch.triu(torch.ones(context_length,context_length),diagonal=1).bool()
output = output.masked_fill(mask,float('-inf'))
# print(pd.DataFrame(output[0,0].detach().numpy()))

## softmax
attentionn_score = F.softmax(output,dim=1)

## attention @ V
A = attentionn_score @ V

## concateate
# A = A.permute(0,2,1,3).reshape(batch_size,context_length,d_modal)
A = A.transpose(1,2).reshape(batch_size,-1,d_modal)
Wo = nn.Linear(d_modal,d_modal)
output = Wo(A)

## residual connection 残差连接
x = x + output

# Layer Norm
layer_norm = nn.LayerNorm(d_modal)
layer_norm_output = layer_norm(output)

# Feed Forward
output = nn.Linear(d_modal,d_modal * 4)(layer_norm_output) # 放大维度
output = nn.ReLU()(output)                      # 激活函数处理
output = nn.Linear(d_modal * 4,d_modal)(output)

# residual connection 残差连接
output = output + layer_norm_output

# Layer Norm
output = layer_norm(output)



# Linear Layer
output = nn.Linear(d_modal,max_token_value+1)(output)
print(output.shape) # torch.Size([4, 16, 199854])

# Softmax
logits = F.softmax(output,dim=-1)