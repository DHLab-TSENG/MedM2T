import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional = True, return_last_step = True):
        """
        LSTM Model
        Args:
            input_size (int): Number of input features (E)
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            bidirectional (bool, optional): If True, use BiLSTM. Default: True
            output_size (int, optional): Number of classes for classification tasks. If None, no classification head.
        """
        super(LSTM, self).__init__()
        self.return_last_step = return_last_step
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (Tensor): Input tensor of shape (B, T, E)
        Returns:
            out (Tensor): 
                - If output_size is None: Hidden states of shape (B, T, hidden_size * 2)
                - Otherwise: Logits of shape (B, output_size)
        """
        self.lstm.flatten_parameters()
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        if self.return_last_step:
            out = out[:, -1, :]

        return out
    

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,  bidirectional = True, return_last_step = True):
        """
        GRU Model
        Args:
            input_size (int): Number of input features (E)
            hidden_size (int): Number of hidden units in GRU
            num_layers (int): Number of GRU layers
            bidirectional (bool, optional): If True, use BiGRU. Default: True
            output_size (int, optional): Number of classes for classification tasks. If None, no classification head.
        """
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Define GRU layer
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        self.return_last_step = return_last_step

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (Tensor): Input tensor of shape (B, T, E)
        Returns:
            out (Tensor): 
                - If output_size is None: Hidden states of shape (B, T, hidden_size * 2)
                - Otherwise: Logits of shape (B, output_size)
        """
        # Initialize hidden state (h0)
        self.gru.flatten_parameters()
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: (B, T, hidden_size * 2), hidden: (num_layers * 2, B, hidden_size)

        if self.return_last_step:
            out = out[:, -1, :]
        
        return out