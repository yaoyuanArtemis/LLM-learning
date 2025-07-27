import torch     # 2.7.1
import torch.nn as nn
import torch.nn.functional as F
import os
import requests  # 2.32.4
import tiktoken
import pandas as pd
import math


## hyperparameter
context_length = 16 
d_modal = 64
batch_size = 4
num_heads = 4
num_blocks = 8
learning_rate = 1e-3
dropout = 0.1
max_iters = 500
eval_interval = 50
eval_iters = 20
device = "cuda" if torch.cuda.is_available() else 'cpu'
TORCH_SPEED = 1337
torch.manual_seed(TORCH_SPEED)

## 请求数据集
if not os.path.exists("data/sales-textbooks.txt"):
    url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true"
    with open("data/sales-textbooks.txt","w") as f:
        f.write(requests.get(url).text)

with open("sales-textbooks.txt","r") as f:
    text = f.read()
    # print("🚀 text:",text)

encoding = tiktoken.get_encoding("o200k_base")
tokenized_text = encoding.encode(text)
tokenized_text = torch.tensor(tokenized_text,dtype=torch.long)
max_token_value = tokenized_text.max().item() + 1 # note!


## split train_set and validation_set
train_index = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_index]
validate_data = tokenized_text[train_index:]

class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_modal,d_modal * 4)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(d_modal * 4,d_modal)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ScaledAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_modal,d_modal)
        self.Wk = nn.Linear(d_modal,d_modal)
        self.Wv = nn.Linear(d_modal,d_modal)
        self.register_buffer("mask",torch.tril(torch.ones(context_length,context_length)))
    
    def DotProduct(self,x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        attention = Q @ K.transpose(-2,-1) / math.sqrt(d_modal // num_heads)
        attention = attention.masked_fill(self.mask == 0,float("-inf"))
        attention = F.softmax(attention,dim=-1)
        attention = attention @ V

class MultiScaledAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledAttention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_modal,d_modal)    
        self.dropout = nn.Dropout()                       
                                   

    def forward(self,x):
        self.heads = [head(x) for head in self.heads]
        out = torch.cat(self.heads,dim=-1)
        out = self.projection_layer(out)
        out = self.dropout(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_modal)
        self.layer_norm2 = nn.LayerNorm(d_modal)
        self.multi_head_attention = MultiScaledAttention()
        self.feedforward_network = FeedforwardNetwork()

    def forward(self,x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))
        return x
    
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_lookup_table = nn.Embedding(max_token_value,d_modal)
        self.transformer_blocks = nn.ModuleList([TransformerBlock() for _ in range(num_blocks)])
        self.linear = nn.Linear(d_modal,max_token_value)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        position_encoding_lookup_table = torch.zeros(context_length,d_modal,device=device)
        position = torch.arange(0,context_length,dtype=torch.float).unsqueeze(1)
        div_item = torch.exp(torch.arange(0,d_modal,2).float() * (-math.log(10000.0) / d_modal))
        position_encoding_lookup_table[:,0::2] = torch.sin(position * div_item)
        position_encoding_lookup_table[:,1::2] = torch.cos(position * div_item)
        position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size,-1,-1)
        position_embedding = position_encoding_lookup_table[:T,:].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B,T,C = logits.shape
            logits_reshaped = logits.view(B * T,C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped,target=targets_reshaped)
        else:
            loss = None
        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length:]
            logits, loss = self(idx_crop)
            logits_last_timestep = logits[:, -1, :]
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = TransformerLanguageModel().to(device)

def get_batch(split: str):
    data = train_data if split == 'train' else validate_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model-ckpt.pt')

model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')


