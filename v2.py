import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # the number of independent sequences to be processed in parallel
block_size = 8 # the maximum context length for predictions
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 32
# ---

torch.manual_seed(1337)


# **Import the initial training data (Shakespeare's works)**

# !wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()


# **Extract Characters**
chars = sorted(list(set(text)))
vocab_size = len(chars)

# **Create encoding/decoding**
stoi = { ch:i for i,ch in enumerate(chars) } #fun fact: stoi = C++ library for converting strings -> integers.
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encodes characters in a string as an array of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decodes integer array back to characters

# **Split out training and validation data sets**
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9* len(data)) # Split out 90% of the data for training
train_data = data[:n]
val_data = data[n:]


# **Data Loading**
def get_batch(split):
    # Generate a small batch of data of input x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Get batch_size random integers between 0 and the data
    x = torch.stack([data[i : i + block_size] for i in ix]) # 4 rows of 8 random encoded character sets
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix]) # the same as the above, offset by one (targets)
    return x, y

@torch.no_grad()
# This decorator turns off gradient calculations, because we don't need these intermediate variables, and won't need to do backward propagation
def estimate_loss():
    out = {}
    model.eval() # Turn on evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Turn on training mode
    return out

# **Super Simple Bigram Language Model**
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.token_position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape # Batch size and time steps

        # idx and targets are both (B, T) tensor or integers
        token_emb = self.token_embedding_table(idx) # (Batch [4] by Time [8] by Channel [n_embed] tensor)
        pos_emb = self.token_position_embedding_table(torch.arange(T, device=device)) # T by C
        x = token_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (Batch by Time by vocab_size)
        loss = None

        if targets is not None:
            # Convert dimenions from B, T ,C -->  B, C, T so that we can use F.cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Negative loglikelihood loss
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# **Create and test a basic Optimiser**

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

batch_size = 32
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step iter {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))