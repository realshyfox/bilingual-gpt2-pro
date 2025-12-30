"""
GPT-2 Base Model Architecture
Transformer decoder-only model for language generation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Config:
    """Configuration for GPT-2 model."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_flash_attention: bool = False,
        position_encoding: str = 'learned',
    ):
        """
        Initialize GPT-2 configuration.
        
        Args:
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_flash_attention: Use Flash Attention 2
            position_encoding: 'learned', 'rope', or 'alibi'
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_flash_attention = use_flash_attention
        self.position_encoding = position_encoding
        
        # Derived values
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, config: GPT2Config):
        """Initialize multi-head attention."""
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        
        # Q, K, V projections
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [batch, num_heads, seq_len, seq_len]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        attn = attn.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        # [batch, num_heads, seq_len, head_dim]
        out = attn @ v
        
        # Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: GPT2Config):
        """Initialize MLP."""
        super().__init__()
        hidden_dim = config.embed_dim * config.mlp_ratio
        
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, config: GPT2Config):
        """Initialize transformer block."""
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections."""
        # Pre-norm attention
        x = x + self.attn(self.ln1(x), attention_mask)
        
        # Pre-norm MLP
        x = x + self.mlp(self.ln2(x))
        
        return x


class GPT2Model(nn.Module):
    """GPT-2 transformer model."""
    
    def __init__(self, config: GPT2Config):
        """
        Initialize GPT-2 model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Position embedding (learned)
        if config.position_encoding == 'learned':
            self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        else:
            self.pos_emb = None
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.embed_dim)
        
        # Language model head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"✅ Initialized GPT-2 model with {self.num_parameters():,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for loss computation [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_emb(input_ids)
        
        # Position embeddings
        if self.pos_emb is not None:
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos_ids)
        
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def num_parameters(self) -> int:
        """Count number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated token IDs [batch, generated_len]
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self.forward(input_ids)
                
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for sequence length limit
                if input_ids.shape[1] >= self.config.max_seq_len:
                    break
        
        return input_ids


def create_model_from_config(config_dict: dict) -> GPT2Model:
    """
    Create GPT-2 model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        GPT-2 model instance
    """
    model_config = config_dict.get('model', {})
    
    config = GPT2Config(
        vocab_size=model_config.get('vocab_size', 50257),
        max_seq_len=model_config.get('max_seq_len', 1024),
        embed_dim=model_config.get('embed_dim', 768),
        num_layers=model_config.get('num_layers', 12),
        num_heads=model_config.get('num_heads', 12),
        dropout=model_config.get('dropout', 0.1),
        attention_dropout=model_config.get('attention_dropout', 0.1),
        use_flash_attention=model_config.get('use_flash_attention', False),
        position_encoding=model_config.get('position_encoding', 'learned'),
    )
    
    model = GPT2Model(config)
    
    return model
