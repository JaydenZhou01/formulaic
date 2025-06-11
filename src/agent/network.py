import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

class NetworkOutputs(NamedTuple):
    pi_prob: torch.Tensor
    value: torch.Tensor

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Independent query per head
        self.querys = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)
        ])
        # Shared key and value
        self.key = nn.Linear(embed_dim, self.head_dim)
        self.value = nn.Linear(embed_dim, self.head_dim)
    
    def self_attention(self, q, k, v):
        # q, k, v: [B, T, D/H]
        k_T = k.transpose(-1, -2)
        attn_scores = torch.matmul(q, k_T) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output

    def forward(self, x):
        B, T, D = x.shape

        # K, V: [B, T, D/H]
        k = self.key(x)
        v = self.value(x)

        attn_output = torch.cat([
            self.self_attention(query(x), k, v) for query in self.querys
        ], dim=-1)
        return attn_output


class MQAEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mqa = MultiQueryAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.mqa(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class RepresentationNet(nn.Module):
    def __init__(self, num_actions: int, embedding_dim: int):
        super().__init__()
        self.num_actions     = num_actions
        self.embedding_dim   = embedding_dim

        # project action one-hot to embedding
        self.action_proj = nn.Linear(num_actions, embedding_dim)

        # stack-height embedding
        self.len_proj = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
        )

        # Multi Query Transformer encoder
        # self.encoder = nn.Sequential(
        #     *[MQAEncoderLayer(embedding_dim, num_heads=4) for _ in range(6)]
        # )

        self.encoder = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True
        )

        # RNN to aggregate encoding into a single embedding
        self.rnn = nn.GRU(embedding_dim, embedding_dim, batch_first=True)

    def forward(self, program: torch.Tensor, stack_len: torch.Tensor):
        """
        program:   [B, T]  LongTensor of token IDs  
        stack_len: [B]     FloatTensor or LongTensor of current stack heights
        """
        B, T = program.shape
        device = program.device

        # 1) one-hot + action_proj → [B, T, D]
        action_1hot = F.one_hot(program, num_classes=self.num_actions).float()
        x = self.action_proj(action_1hot)

        # 2) add stack-height embedding
        #    stack_len: [B] → [B,1] → project → [B, D] → unsqueeze → [B,1,D] → broadcast to [B,T,D]
        sl = stack_len.view(B, 1).float()                        # ensure float
        len_emb = self.len_proj(sl)                              # [B, D]
        len_emb = len_emb.unsqueeze(1).expand(-1, T, -1)         # [B, T, D]
        x = x + len_emb

        # 3) transformer encoding
        # x = self.encoder(x)  # [B, T, D]
        x, _ = self.encoder(x, x, x)

        # 4) aggregate via RNN; take last hidden state as representation
        _, h_n = self.rnn(x)     # h_n: [1, B, D]
        rep = h_n[0]             # [B, D]

        return rep
    
class DistributionSupport:
    def __init__(self, value_size: float, num_bins: int):
        self.value_size = value_size
        self.num_bins = num_bins
        self.bins = torch.linspace(-value_size, value_size, num_bins)

    def mean(self, probs: torch.Tensor):
        return torch.sum(probs * self.bins.to(probs.device), dim=-1)

    def scalar_to_two_hot(self, scalar: torch.Tensor):
        # scalar: shape [B]
        scalar = torch.clamp(scalar, -self.value_size, self.value_size)
        idx = (scalar - (-self.value_size)) / (2 * self.value_size) * (self.num_bins - 1)
        lower = torch.floor(idx).long()
        upper = torch.clamp(lower + 1, 0, self.num_bins - 1)
        weight_upper = idx - lower.float()
        weight_lower = 1.0 - weight_upper

        one_hot_lower = F.one_hot(lower, num_classes=self.num_bins)
        one_hot_upper = F.one_hot(upper, num_classes=self.num_bins)

        return weight_lower.unsqueeze(-1) * one_hot_lower + weight_upper.unsqueeze(-1) * one_hot_upper


class CategoricalHead(nn.Module):
    def __init__(self, embedding_dim: int, value_support: DistributionSupport):
        super().__init__()
        self.value_support = value_support
        self.linear = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, value_support.num_bins)
        )

    def forward(self, x):
        logits = self.linear(x)  # [B, num_bins]
        probs = F.softmax(logits, dim=-1)
        mean = self.value_support.mean(probs)  # [B]
        return {
            'logits': logits,
            'mean': mean
        }


class PredictionNet(nn.Module):
    def __init__(self, embedding_dim: int, num_actions: int,
                 value_size: float = 1.0, value_num_bins: int = 41):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_actions)
        )
        self.value_support = DistributionSupport(value_size, value_num_bins)
        self.value_head = CategoricalHead(embedding_dim, self.value_support)

    def forward(self, embedding: torch.Tensor):
        policy_logits = self.policy_head(embedding)           # [B, num_actions]
        value_output = self.value_head(embedding)             # {'logits': ..., 'mean': ...}
        
        return (policy_logits, value_output['logits'], value_output['mean']) 

class Network(nn.Module):
    def __init__(self, num_actions, embedding_dim):
        super().__init__()
        self.representation_net = RepresentationNet(num_actions, embedding_dim)
        self.prediction_net = PredictionNet(embedding_dim, num_actions)

    def forward(self, program, stack_len):
        embedding = self.representation_net(program, stack_len)
        pred = self.prediction_net(embedding)
        return pred
    
    def inference(self, program, stack_len):
        with torch.no_grad():
            return self.forward(program, stack_len)
        