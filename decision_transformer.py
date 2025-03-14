import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=3*128*128, action_dim=1, embed_dim=128, seq_len=5):
        super(DecisionTransformer, self).__init__()

        self.state_dim = state_dim
        self.embedding = nn.Linear(self.state_dim, embed_dim)

        self.transformer = nn.Transformer(
            embed_dim, nhead=4, num_encoder_layers=2, num_decoder_layers=2, batch_first=True
        )

        self.action_head = nn.Linear(embed_dim, action_dim)

    def forward(self, returns, states, actions):
        batch_size = states.shape[0]

        states = states.view(batch_size, states.shape[1], -1)  
        states = self.embedding(states) 

        # Transformer forward pass
        output = self.transformer(states, states)

        action_prediction = self.action_head(output)  

        return action_prediction.squeeze(-1)  

