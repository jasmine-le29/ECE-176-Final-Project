import torch
import torch.nn as nn
import torchvision.models as models

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=(3, 128, 128), action_dim=1, embed_dim=256, seq_len=5):
        super(DecisionTransformer, self).__init__()

        self.seq_len = seq_len
        self.action_dim = action_dim

        # Pretrained ResNet-18 as Feature Extractor
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False  

        # Compute feature size
        with torch.no_grad():
            dummy_input = torch.randn(1, *state_dim)
            cnn_output_dim = self.resnet(dummy_input).view(-1).shape[0]

        # Linear transformation for embeddings
        self.state_embedding = nn.Linear(cnn_output_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.return_embedding = nn.Linear(1, embed_dim)

        # Transformer with improved depth and heads
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=8,  
            num_encoder_layers=6, 
            num_decoder_layers=6,
            batch_first=True
        )

        # Output layer
        self.action_head = nn.Linear(embed_dim, action_dim)

    def forward(self, returns, states, actions):
        batch_size = states.shape[0]

        # Extract features using ResNet-18
        states = states.view(batch_size * self.seq_len, *states.shape[2:])
        with torch.no_grad():
            states = self.resnet(states)
        states = states.view(batch_size, self.seq_len, -1)
        states = self.state_embedding(states)

        # Embed actions and returns-to-go
        actions = self.action_embedding(actions.unsqueeze(-1))
        returns = self.return_embedding(returns.unsqueeze(-1))

        # Combine embeddings
        x = states + actions + returns  

        # Transformer with causal masking
        seq_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(x.device)
        output = self.transformer(x, x, tgt_mask=seq_mask)

        # Predict steering angle
        predicted_action = self.action_head(output)  

        return predicted_action.squeeze(-1)



