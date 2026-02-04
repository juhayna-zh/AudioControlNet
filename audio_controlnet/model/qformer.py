import torch
from torch import nn
from einops import rearrange

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj


class EventQFormerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qformer = EncoderProjectorQFormer(config)
        self.query_len = self.qformer.query_len
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim

        # Linear layer to aggregate multiple query tokens
        self.query_agg = nn.Linear(self.query_len, 1, bias=False)

    def forward(self, x, num_events):
        """
        Args:
            x: Tensor of shape [B, N, E, T]
            num_events: list[int] indicating valid event counts per batch item
        Returns:
            Tensor of shape [B, T, llm_dim], after aggregating query_len dim
        """
        B, N, E, T = x.shape
        
        # Flatten time dimension into batch dimension: [B, N, E, T] -> [B*T, N, E]
        x = rearrange(x, 'b n e t -> (b t) n e')
        
        # Create attention mask: 1 for valid events, 0 for padding
        mask = torch.zeros(B, T, N, dtype=torch.long, device=x.device)
        for i in range(B):
            mask[i, :, :num_events[i]] = 1
        mask = rearrange(mask, 'b t n -> (b t) n')
        
        # Forward pass through the Q-Former
        out = self.qformer(x, mask)  # [B*T, query_len, llm_dim]
        
        # Apply linear over query_len dim
        out = rearrange(out, '(b t) q l -> b q t l', b=B) # Reshape output: [B*T, query_len, llm_dim] -> [B, query_len, T, llm_dim]
        out = out.permute(0, 2, 3, 1)  # [B, T, llm_dim, query_len]
        out = self.query_agg(out)  #  [B, T, llm_dim, 1]
        out = out.squeeze(-1)  # [B, T, llm_dim]
        
        return out


