import torch     # 2.7.1
import torch.nn as nn
import torch.nn.functional as F
import os
import requests  # 2.32.4



# 如果没有文件 请求数据集
if not os.path.exists("sales-textbooks.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("sales-textbooks.txt","wb") as f:
        f.write(requests.get(url).content)

with open("sales-textbooks.txt","r") as f:
    text = f.read()
    # print("🚀 text:",text)

import tiktoken
encoding = tiktoken.get_encoding("o200k_base")
tokenized_text = encoding.encode(text)
print("tokenized_text: ",tokenized_text)
print(len(tokenized_text))