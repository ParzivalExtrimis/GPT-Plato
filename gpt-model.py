from PyPDF2 import PdfReader
import os
import glob
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
max_iters = 5000
eval_interval = 500
eval_iters = 200
lrate = 3e-4
n_embbed = 384
block_s = 128
batch_s = 64
n_heads = 6
n_layer = 6
dropout_r = 0.2

directory_path = 'datasets\sam_harris_podcast_transcripts'
pdf_files = glob.glob(os.path.join(directory_path, '*.pdf')) # get all dataset chunks

#get the total size of dataset
text = ''
for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    pages = reader.pages

    # extracting text from page
    for page in pages:
        text += page.extract_text()

print('Chars: ', len(text))

#make vocab
vocab = sorted(list(set(text)))
vocab_s = len(vocab)

print('Vocab: ', vocab)
print('Vocab size: ', vocab_s)

#make encoder/ decoder
#   make stoi, itos dicts to hold translations

itos = {i : s for i, s in zip(range(vocab_s), vocab)}
stoi = {s : i for i, s in itos.items()}

def encode(in_str):
    return [stoi[c] for c in in_str]

def decode(in_int_list):
    return ''.join([itos[x] for x in in_int_list])

train_split_n = 0.9 # 90% of the dataset used in training.
val_split_n = 0.1 # 10% of the dataset used in validation.

# encode text (data set) -> data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_split_n * len(data))
train_split = data[: n]
val_split = data[n :]

# make a func to take in data in splits and return Xs and Ys
def build_batch(split):
    #determine split type and use relevant set
    dta = train_split if split == 'train' else val_split
    # sample block size examples at random from set
    ix = torch.randint(0, len(dta) - block_s, (batch_s, ))
    xs = torch.stack([dta[i : i + block_s] for i in ix])
    ys = torch.stack([dta[i + 1 : i + block_s + 1] for i in ix])
    xs, ys = xs.to(device), ys.to(device)
    return xs, ys

xs, ys = build_batch('train')

#eval loss function
@torch.no_grad()
def get_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = build_batch(split)
            logits , loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class AttentionHead(torch.nn.Module):
    def __init__(self, head_s):
        super().__init__()
        self.head_s = head_s

        #initialize K, Q, V, tril
        self.key = torch.nn.Linear(n_embbed, head_s, bias=False)
        self.query = torch.nn.Linear(n_embbed, head_s, bias=False)
        self.value = torch.nn.Linear(n_embbed, head_s, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_s, block_s)))

        self.dropout = torch.nn.Dropout(dropout_r)

    def forward(self, x):
        B, T, E = x.shape
        k = self.key(x) # (B, T, E) @ (E, H) -> (B, T, H)
        q = self.query(x) # (B, T, H)

        #get attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5) # (B, T, T) # /sqrt(head_S) to normalize the initalizations of heads
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)# section the matrix such that nodes can not see their children 
        wei = torch.softmax(wei, dim=1)# (B, T, T)
        wei = self.dropout(wei)

        #weighed aggrigation of the values
        v = self.value(x) # ( B, T, E ) @ ( E, H ) -> (B, T, H)
        out = wei @ v # ( B, T, T ) @ ( B, T, H ) -> ( B, T, H )
        return out
    
class MultiheadAttention(torch.nn.Module):
    def __init__(self, n_heads, size):
        super().__init__()

        #initalize each head as a module list
        self.heads = torch.nn.ModuleList(AttentionHead(size) for _ in range(n_heads))
        self.proj = torch.nn.Linear(size * n_heads, n_embbed) # allows for the output to fork back into residual pathway
        self.dropout = torch.nn.Dropout(dropout_r)

    def forward(self, x):
        #call each head in list sequentially and concat the outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1) # h(B, T, H)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        
        #init a sequential linear-> non-linear MLP
        hidden_s = size * 4 # increases compute dimension of ffwd
        self.net = torch.nn.Sequential(
            torch.nn.Linear(size, hidden_s),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_s, size),# Projection: allows for the output to fork back into residual pathway
            torch.nn.Dropout(dropout_r)
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embbed, n_heads):
        super().__init__()
        
        #initalize both multihead and ffwds
        head_s = n_embbed // n_heads
        self.mh = MultiheadAttention(n_heads, head_s)
        self.ffwd = FeedForward(n_embbed)
        self.attention_layer_norm = torch.nn.LayerNorm(n_embbed)
        self.ffwd_layer_norm = torch.nn.LayerNorm(n_embbed)

    def forward(self, x):
        x = x + self.mh(self.attention_layer_norm(x))
        x = x + self.ffwd(self.ffwd_layer_norm(x))
        return x
    
class LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        #initialize emb tables
        self.token_embedding_table = torch.nn.Embedding(vocab_s, n_embbed) # (vocab_s, embedding_dim_s)
        self.pos_embedding_table = torch.nn.Embedding(block_s, n_embbed)

        #layers
        self.blocks = torch.nn.Sequential(*[Block(n_embbed, n_heads=n_heads) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embbed) # final layer norm
        self.lm_head = torch.nn.Linear(n_embbed, vocab_s)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embbeds = self.token_embedding_table(idx) # (B, T, E) { E = embedding_dim_s}
        pos_embbeds = self.pos_embedding_table(torch.arange(T, device=device)) # (T, E)
        embbeds = token_embbeds + pos_embbeds # (B, T, E)
        x = self.blocks(embbeds) # (B, T, E)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, C) 
        
        # during generation targets, loss is not req
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # (BT, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_gens):
        for _ in range(max_gens):
            #get logits from forward pass use the last character
            #pass context_length(idx) i.e len <= block_size { position embedding in fwd only defined upto block_s}
            context = idx[:, -block_s: ]
            logits, _ = self(context) # (B, T, C) (b, t) (b, t, C)
            logits = logits[:, -1, :] # (b, C)
            #softmax logits to get probabilities 
            probs = torch.nn.functional.softmax(logits, dim=-1) # (b, C)
            #use probs to sample from multinomial and append to the idx
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # # (b, C + 1)
        return idx
    
model = LanguageModel()
m = model.to(device)

# log the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)


for iter in range(max_iters):

    #display loss
    if iter % eval_interval == 0:
        losses = get_loss()
        print(f'{iter} Train Loss: {losses["train"]}, Validation Loss: {losses["val"]}')

    #get batch from batch sampler
    x, y = build_batch('train')
    
    #foward pass
    logits, loss = model(x, y)
    
    #backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    #optimize
    optimizer.step()
    
print(loss)

#sample generator
context = torch.zeros((1, 1), dtype=torch.long, device=device)
idx = m.generate(context, 200)[0].tolist()
out = decode(idx)
print(out)