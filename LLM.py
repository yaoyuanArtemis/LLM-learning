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
# print("tokenized_text: ",tokenized_text)


# split train_set and validation_set
train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validate_data = tokenized_text[train_index:]

# hyperparameter
context_length = 16 
d_modal = 64
bach_size = 4
