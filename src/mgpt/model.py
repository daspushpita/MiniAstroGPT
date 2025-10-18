import torch
import torch.nn as nn
import torch.nn.functional as F

class GenerateEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        positional_embeddings = self.position_embedding(position_ids)
        return token_embeddings + positional_embeddings
    

class CausalAttention(nn.Module):
    
    def __init__(self, embed_dim, context_length, dropout, qkv_bias=False, *args, **kwargs):
        super(CausalAttention, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.W_query = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.W_key = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.W_value = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        
        self.context_length = context_length
        self.register_buffer("mask", 
                             torch.triu(torch.ones(self.context_length, self.context_length), 
                            diagonal=1))
        
    def forward(self, x):
        batch_size, seq_length, d_model = x.shape   #x is of shape (batch_size, seq_length, embed_dim)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:seq_length, :seq_length], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, context_length, num_heads, dropout, qkv_bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_query = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.W_key = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.W_value = nn.Linear(self.embed_dim, self.embed_dim, bias= qkv_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.register_buffer("mask",
                            torch.triu(torch.ones(self.context_length, self.context_length),
                            diagonal=1)
        )
    
    def forward(self,x):
        batch_size, seq_length, d_model = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:seq_length, :seq_length]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        scale = keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.last_attn = attn_weights.detach().cpu() #For debugging
        
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, seq_length, self.embed_dim)
        
        context_vec = self.out_proj(context_vec)
        
        return context_vec
    

class FeedForward(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.neural_net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        
    def forward(self, x):
        return self.neural_net(x)
    
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1.e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean)/torch.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta

class TransformerBlock(nn.Module):
    def __init__(self, params_dict):
        super().__init__()
        self.attention = MultiHeadAttention(params_dict["embed_dim"], params_dict["context_dim"], params_dict["num_heads"], params_dict["dropout"])
        self.layernorm1 = LayerNorm(params_dict["embed_dim"])
        self.ffn = FeedForward(params_dict["embed_dim"])
        self.layernorm2 = LayerNorm(params_dict["embed_dim"])
        self.drop_shortcut = nn.Dropout(params_dict["dropout"])
        
    def forward(self, x):
        shortcut = x
        x = self.layernorm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
class Mini_AstroGPT_Model(nn.Module):
    def __init__(self, params_dict):
        super().__init__()
        self.embed_layer = GenerateEmbeddings(params_dict["vocab_size"], params_dict["embed_dim"], params_dict["context_dim"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(params_dict) for _ in range(params_dict["num_layers"])])
        self.final_norm = LayerNorm(params_dict["embed_dim"])
        self.out_head = nn.Linear(params_dict["embed_dim"], params_dict["vocab_size"], bias=False)
        self.out_head.weight = self.embed_layer.token_embedding.weight
        
    def forward(self, x):
        x = self.embed_layer(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
