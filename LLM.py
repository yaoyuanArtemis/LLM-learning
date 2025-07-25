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
# print("tokenized_text: ",tokenized_text)


# split train_set and validation_set
train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validate_data = tokenized_text[train_index:]

# hyperparameter
context_length = 16 
d_modal = 64
bach_size = 4

# 训练数据中整批次整理好数据
data = train_data
idxs = torch.randint(low=0,high=len(data) - context_length,size=(bach_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])
y_batch = torch.stack([data[idx+1:idx + context_length+1] for idx in idxs])
print(x_batch)

import pandas as pd
# print(pd.DataFrame(x_batch[0].numpy()))

# create Input Embedding Table
max_token_value = tokenized_text.max().item()
input_embedding_lookup_table = nn.Embedding(max_token_value+1,d_modal)
print("input_embedding_lookup_table",input_embedding_lookup_table)
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch)

# positional encoding

