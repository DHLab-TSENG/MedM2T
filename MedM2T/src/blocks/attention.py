import torch.nn as nn

class CrossAttn(nn.Module):
    """
    Cross Attention Block
    Args:
        embed_size (int): Embedding size
        num_heads (int): Number of heads
        drop_prob (float): Dropout probability
    """
    def __init__(self,embed_size, num_heads, drop_prob):
        super(CrossAttn, self).__init__()

        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(embed_size, 4 * embed_size),
                                  nn.LeakyReLU(),
                                  nn.Linear(4 * embed_size, embed_size))
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Input tensor 1 (B, E, T)
            x2 (torch.Tensor): Input tensor 2 (B, E, T)
        Returns:
            torch.Tensor: Output tensor (B, E, T)
        """
        attn_out, _ = self.attention(x1, x2, x2, need_weights=False)
        x1 = x1 + self.dropout(attn_out)
        x1 = self.ln1(x1)

        fc_out = self.fc(x1)
        x1 = x1 + self.dropout(fc_out)
        x1 = self.ln2(x1)
        return x1
    

class FeatureSelfAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_prob, out_dim):
        """
        Feature Self Attention Block
        Args:
            embed_dim (int): Embedding size
            num_heads (int): Number of heads
            drop_prob (float): Dropout probability
            out_dim (int): Output dimension
        """
        super(FeatureSelfAttn, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.fc = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(4 * embed_dim, embed_dim))
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.final_fc = nn.Linear(embed_dim, out_dim)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, E, T)
        Returns:
            torch.Tensor: Output tensor (B, E, T)
        """
        #(B, E, T)
        x = x.permute(2, 0, 1)#(T, B, E)
        
        attn_out, _ = self.self_attention(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)
        x = self.final_fc(x)

        #(B, E, T)
        x = x.permute(1, 2, 0)
        return x