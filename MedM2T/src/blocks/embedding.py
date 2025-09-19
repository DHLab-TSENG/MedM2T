import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
        
class TimeWinEmbedding(nn.Module):
    def __init__(self, value_vocab_size, source_vocab_size, win_size, embed_dim, device, 
                 temporal_weighted = True, shared_embedding = False):
        """
        Time Window Embedding
        Args:
            value_vocab_size (int): Number of unique values in the value vocabulary
            source_vocab_size (int): Number of unique values in the source vocabulary
            win_size (int): Number of time windows
            embed_dim (int): Embedding dimension
            device (torch.device): Device to run the model
            temporal_weighted (bool, optional): If True, apply temporal weighting to the embeddings. Default: True
            shared_embedding (bool, optional): If True, share the embedding across time windows. Default: False
        """
        super(TimeWinEmbedding, self).__init__()
        self.win_size = win_size
        self.device = device
        self.shared_embedding = shared_embedding
        if self.shared_embedding:
            self.win_value_embedding = nn.Embedding(value_vocab_size, embed_dim)
            self.win_source_embedding = nn.Embedding(source_vocab_size, embed_dim)
        else:
            self.win_value_embedding = nn.ModuleList([nn.Embedding(value_vocab_size, embed_dim) for _ in range(win_size)])
            self.win_source_embedding = nn.ModuleList([nn.Embedding(source_vocab_size, embed_dim) for _ in range(win_size)])
        
        self.temporal_weighted = temporal_weighted
        if temporal_weighted:
            self.win_weight = nn.Parameter(torch.randn(win_size)) 
        else:
            self.win_weight = None

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (tuple): Tuple of two tensors
                - x[0] (tuple): Tuple of two tensors representing values in the time windows
                    - x[0][0] (torch.Tensor): List of value tokens in the time windows (T, V)
                    - x[0][1] (torch.Tensor): List of token sizes in the time windows (T, B), representing the token size for each data point Xi
                - x[1] (tuple): Tuple of two tensors representing sources in the time windows
                    - x[1][0] (torch.Tensor): List of source tokens in the time windows (T, V)
                    - x[1][1] (torch.Tensor): List of token sizes in the time windows (T, B), representing the token size for each data point Xi
        Returns:
            torch.Tensor: Output tensor (B, T, E) if temporal_weighted is False, otherwise (B, E)
        """
        win_values = x[0][0]
        win_sources = x[1][0]
        win_tokens_size = x[0][1] # same as x[1][1]
        win_embs = []
        for win_i in range(self.win_size):
            vals = win_values[win_i]
            srcs = win_sources[win_i]
            tokens_size = win_tokens_size[win_i]
            if self.shared_embedding:
                v_embs = self.win_value_embedding(vals)
                s_embs = self.win_source_embedding(srcs)
            else:
                v_embs = self.win_value_embedding[win_i](vals)
                s_embs = self.win_source_embedding[win_i](srcs)

            batch_indices = torch.arange(len(tokens_size)).repeat_interleave(tokens_size.cpu()).to(self.device)
            vs_embs = v_embs * s_embs
            win_emb = scatter_mean(vs_embs, batch_indices, dim=0)
            win_embs.append(win_emb)
        
        win_embs = torch.stack(win_embs, dim = 1)

        if self.temporal_weighted:
            T = win_embs.shape[1]
            emb_weighted = win_embs
            emb_weighted = emb_weighted * self.win_weight.view(1,T,1)
            emb_weighted = emb_weighted.sum(dim = 1)
            return emb_weighted
        else:
            return win_embs

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Embedding, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_size, sparse=False)
        
    def forward(self, x):
        return self.embedding(x[0], x[1])