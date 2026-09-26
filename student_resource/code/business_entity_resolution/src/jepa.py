import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class CharTokenizer:
    def __init__(self):
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789 &,-./"
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.pad_idx = 0
        self.vocab_size = len(self.chars) + 1
        
    def encode(self, text, max_len=64):
        text = str(text).lower()
        seq = [self.char2idx.get(c, 0) for c in text][:max_len]
        seq = seq + [self.pad_idx] * (max_len - len(seq))
        return seq

class JEPADataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def corrupt(self, text):
        if random.random() < 0.3:
            words = text.split()
            if len(words) > 1:
                words.pop(random.randint(0, len(words)-1))
                text = " ".join(words)
        if random.random() < 0.2:
            text = text.replace(' ', '')
        return text
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        orig = str(self.texts[idx])
        v1 = self.corrupt(orig)
        v2 = self.corrupt(orig)
        return (torch.tensor(self.tokenizer.encode(v1, self.max_len)),
                torch.tensor(self.tokenizer.encode(v2, self.max_len)))

class JEPAEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = F.normalize(self.fc(x), dim=-1)
        return x

class JEPAModel(nn.Module):
    def __init__(self, encoder, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        return p1, z2

def train_jepa(texts, epochs=10, batch_size=512):
    print(f"Training JEPA representation model on {len(texts)} records...")
    tokenizer = CharTokenizer()
    dataset = JEPADataset(texts, tokenizer)
    
    # Unleashed: No subsetting! Train on full dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    encoder = JEPAEncoder(tokenizer.vocab_size)
    model = JEPAModel(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Prioritize CUDA for RTX 6000
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU Detected: Using CUDA (RTX 6000)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("GPU Detected: Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("No GPU detected: Using CPU")
        
    model.to(device)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for b_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            p1, z2 = model(x1, x2)
            loss = 1 - F.cosine_similarity(p1, z2.detach()).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"JEPA Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return encoder.to('cpu'), tokenizer

def get_jepa_embeddings(encoder, tokenizer, texts, batch_size=1024):
    encoder.eval()
    embeddings = []
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    encoder.to(device)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = [tokenizer.encode(t) for t in batch]
        x = torch.tensor(encoded).to(device)
        with torch.no_grad():
            emb = encoder(x)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)
