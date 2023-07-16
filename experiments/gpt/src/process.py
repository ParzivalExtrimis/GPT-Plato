import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import mlflow
from azureml.core import Run


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--batch_size", type=int, default=96, required=False, help="")
    parser.add_argument("--block_size", type=int, default=256, required=False, help="")
    parser.add_argument("--epochs", type=int, default=5000, required=False, help="")
    parser.add_argument("--eval_interval", type=int, default=100, required=False, help="")
    parser.add_argument("--eval_iters", type=int, default=200, required=False, help="")
    parser.add_argument("--n_embbed", type=int, default=384, required=False, help="")
    parser.add_argument("--n_heads", type=int, default=6, required=False, help="")
    parser.add_argument("--n_layers", type=int, default=6, required=False, help="")
    parser.add_argument("--dropout", type=float, default=0.2, required=False, help="")
    parser.add_argument("--test_train_ratio",type=float, required=False, default=0.9, help="Specify the ratio (0.9) for the train split")
    parser.add_argument("--learning_rate", required=False, default=3e-4, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    run = Run.get_context()
    run_id = run._run_id
    if isinstance(run_id, str):
        mlflow.start_run(run_id=run_id)
    else:
        mlflow.start_run()

    # hyperparameters
    batch_size = args.batch_size # independent sequences to process in parallel?
    block_size = args.block_size # maximum context length for predictions
    test_train_split_ratio = args.test_train_ratio
    max_iters = args.epochs
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = args.eval_iters
    n_embd = args.n_embbed
    n_head = args.n_heads
    n_layer = args.n_layers
    dropout = args.dropout
    # ------------


    torch.manual_seed(1337)

    with open(args.data, 'r', encoding='latin-1') as f:
        text = f.read()

    log_f = os.makedirs('outputs/logs', exist_ok=True)
    mlflow.log_text(f'Text retrieved from datastore: {text[:40]}', log_f)

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(test_train_split_ratio*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    mlflow.log_metric("Number of tokens:", vocab_size)
    mlflow.log_metric("Dataset size: ", data.shape)

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            mlflow.log_metric('Training Loss', float(out['train']))
            mlflow.log_metric('Validation Loss', float(out['val']))
        model.train()
        return out

    class Head(nn.Module):
        """ one head of self-attention """

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # input of size (batch, time-step, channels)
            # output of size (batch, time-step, head size)
            B,T,C = x.shape
            k = self.key(x)   # (B,T,hs)
            q = self.query(x) # (B,T,hs)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(x) # (B,T,hs)
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out

    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedFoward(nn.Module):
        """ a simple linear layer followed by a non-linearity """

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """ Transformer block: communication followed by computation """

        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPTLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd) # final layer norm
            self.lm_head = nn.Linear(n_embd, vocab_size)

            # better init, not covered in the original GPT video, but important, will cover in followup video
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            # idx and targets are both (B,T) tensor of integers
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
            x = self.blocks(x) # (B,T,C)
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx

    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

    ##########################
    #<save and register model>
    ##########################
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.pytorch.log_model(
        pytorch_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=os.path('outputs', args.registered_model_name),
    )

    # Saving the model to a file
    model_save_path = os.path('outputs', run.experiment.name)
    os.makedirs(model_save_path, exist_ok=True)
    mlflow.pytorch.save_model(
        pytorch_model=model,
        path=model_save_path,
    )
    ###########################
    #</save and register model>
    ###########################

    # Stop Logging
    mlflow.end_run()