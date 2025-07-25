import torch     # 2.7.1
import torch.nn as nn
import torch.nn.functional as F
import os
import requests  # 2.32.4



# 请求数据集
if not os.path.exists("sales-textbooks.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("sales-textbooks.txt","wb") as f:
        f.write(requests.get(url).content)

with open("sales-textbooks.txt","r") as f:
    text = f.read()
    # print("🚀 text:",text)


# tokenize化
import tiktoken
encoding = tiktoken.get_encoding("o200k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text,dtype=torch.long)


# split train_set and validation_set
train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validate_data = tokenized_text[train_index:]

# hyperparameter
context_length = 16 
d_modal = 64
batch_size = 4

# 训练数据中整批次整理好数据
data = train_data
idxs = torch.randint(low=0,high=len(data) - context_length,size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])
y_batch = torch.stack([data[idx+1:idx + context_length+1] for idx in idxs])

import pandas as pd
# print(pd.DataFrame(x_batch[0].numpy()))

# create Input Embedding Table
max_token_value = tokenized_text.max().item()
input_embedding_lookup_table = nn.Embedding(max_token_value+1,d_modal)
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch) # y_batch_embedding.shape = (4 * 16 * 64)

# positional encoding
import math
positinal_encoding_lookup_table = torch.zeros(context_length,d_modal)
position = torch.arange(0,context_length,dtype=torch.float).unsqueeze(1)
div_item = torch.exp(torch.arange(0,d_modal,2).float() * (-math.log(10000.0) / d_modal))

positinal_encoding_lookup_table[:,0::2] = torch.sin(position * div_item)
positinal_encoding_lookup_table[:,1::2] = torch.cos(position * div_item)
positinal_encoding_lookup_table = positinal_encoding_lookup_table.unsqueeze(0).expand(batch_size,-1,-1)

x = x_batch_embedding + positinal_encoding_lookup_table
y = y_batch_embedding + positinal_encoding_lookup_table

