import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, num_class, hidden_dim, drop_prob, BatchNorm = True):
        """
        MLP Decoder
        Args:
            in_dim (int): Input dimension
            num_class (int): Number of classes
            hidden_dim (list): List of hidden layer dimensions
            drop_prob (float): Dropout probability
            BatchNorm (bool, optional): If True, apply Batch Normalization. Default: True
        """
        super(MLPDecoder, self).__init__()
        layers = []
        for out_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, out_dim))
            if BatchNorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prob))
            in_dim = out_dim
            
        layers.append(nn.Linear(in_dim, num_class))
        layers.append(nn.LogSoftmax(dim=1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, E)
        Returns:
            torch.Tensor: Output tensor (B, C)
        """
        return self.mlp(x)
    
class MLPDecoderReg(nn.Module):
    """
    MLP Decoder for Regression
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        hidden_dim (list): List of hidden layer dimensions
        drop_prob (float): Dropout probability
        BatchNorm (bool, optional): If True, apply Batch Normalization. Default: True
    """
    def __init__(self, in_dim, out_dim, hidden_dim, drop_prob, BatchNorm = True):
        super(MLPDecoderReg, self).__init__()
        layers = []
        for _out_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, _out_dim))
            if BatchNorm:
                layers.append(nn.BatchNorm1d(_out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prob))
            in_dim = _out_dim
            
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, E)
        Returns:
            torch.Tensor: Output tensor (B, C)
        """
        return self.mlp(x)
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_prob, BatchNorm = True):
        """
        MLP
        Args:
            in_dim (int): Input dimension
            hidden_dim (list): List of hidden layer dimensions
            drop_prob (float): Dropout probability
            BatchNorm (bool, optional): If True, apply Batch Normalization. Default: True
        """
        super(MLP, self).__init__()
        
        layers = []
        for out_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, out_dim))
            if BatchNorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prob))
            in_dim = out_dim
            
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, E)
        Returns:
            torch.Tensor: Output tensor (B, E)
        """
        return self.mlp(x)