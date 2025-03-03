from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import SOS, EOS, CharTokenizer

__version__ = "1.1.1"

@dataclass
class TransformerConfig:
    """
    """
    vocab_size: int
    d_model: int # D or d_model in comments
    n_layers: int
    n_heads: int
    max_len: int # maximum sequence length (for positional embedding)
    dropout: float = 0.1
    bias: bool = False
    norm_eps: float = 1e-5
    super_attn: bool = False # overwrites flash to False
    flash: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be a multiple of n_heads ({self.n_heads})"

        self.d_head = self.d_model // self.n_heads


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim: int, eps: float) -> None:
        """
        Args :
            dim (int) : 
            eps (float) :
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SelfAttentionMultiHead(nn.Module):
    """
    """
    def __init__(self, config: TransformerConfig) -> None:
        """
        Args :
            config (TransformerConfig) :
        """

        super().__init__()

        self.config = config

        # key, query, value projections for all heads
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False) # d_query = n_heads*d_head as in the Transformer paper
        self.key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        if not config.flash:
            # compute the mask once and for all here 
            # registrer treats it like a parameter (device, state_dict...) without training
            mask = torch.full((1, 1, config.max_len, config.max_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)

        # LxL super attention params
        if config.super_attn:
            self.k_in_v_proj = nn.Linear(in_features=config.max_len, out_features=config.max_len, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args :
            x (torch.Tensor) : input shaped (B, S, d_model)
        """
        B, S, _ = x.size()

        Q = self.query_proj(x).view(B, S, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, S, d_query)
        K = self.key_proj(x).view(B, S, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, S, d_key)
        V = self.value_proj(x).view(B, S, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, S, d_head=d_value)

        if self.config.flash and not self.config.super_attn:
            attention = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True
                )
        else:
            QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, S, S)
            QK_T = QK_T + self.mask[:, :, :S, :S]

            attention_scores = torch.softmax(QK_T / math.sqrt(self.config.d_head), dim=3) # (B, n_heads, S, S)

            if self.config.super_attn:
                attention = self.attn_dropout(attention_scores) @ self.k_in_v_proj.weight @ V # (B, n_h, L, d_value=d_head)
            else:
                attention = self.attn_dropout(attention_scores) @ V # (B, n_h, S, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, S, n_heafs, d_head)
        y = attention.contiguous().view(B, S, self.config.d_model) # n_heads * d_head = d_model

        y = self.resid_dropout(self.c_proj(y))

        return y # (B, S, d_model)
    

class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        Args :
            config (TransformerConfig) : configuration settings
        """
        super().__init__()

        self.fc_1 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.fc_2 = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.fc_3 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args :
            x (torch.Tensor) : input data  shaped (B, S, d_model)
        """
        x = self.dropout(self.fc_2(F.silu(self.fc_1(x)) * self.fc_3(x)))
        return x # (B, S, d_model)
    

class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        Args :
            config (TransformerConfig) :
        """
        super().__init__()

        self.config = config

        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.sa = SelfAttentionMultiHead(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
         Args :
            x (torch.Tensor) : input data shaped (B, S, d_model)
        """
        x = x + self.sa(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x # (B, S, d_model)
    

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        """
        super().__init__()

        self.config = config

        # Positional Embedding
        self.PE = nn.Embedding(config.max_len, config.d_model) 
        self.in_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, x: torch.Tensor, stop_at_layer: int = -1) -> torch.Tensor:
        """
        Args :
            x (torch.Tensor) : input data shaped (B, S, d_model)
            stop_at_layer (int) : return the ouput (activations) after the specified {layer}-th layer (1 -> n_layers)
        """
        if stop_at_layer < 0:
            stop_at_layer += len(self.layers) + 1
        elif stop_at_layer == 0:
            stop_at_layer = 1
        assert stop_at_layer <= len(self.layers), \
            f"stop_at_layer ({stop_at_layer}) should be in [{-len(self.layers)}, {len(self.layers)}]"
        _, S, _ = x.size()

        # Add positional embedding
        pos_emb = self.PE(torch.arange(0, S, dtype=torch.long, device=x.device))
        x = self.in_dropout(x + pos_emb)

        for i, layer in enumerate(self.layers):
            x = layer(x) # (B, S, d_model)

            if stop_at_layer == i+1:
                return x
        
        return x # (B, S, d_model)


class LanguageModel(nn.Module):
    def __init__(self, model_config: TransformerConfig) -> None:
        super().__init__()

        self.config = model_config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model, padding_idx=0)
        
        self.core = Transformer(self.config)
        self.out_norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        self._init_normal()

    def _init_weights(self, module):
        # taken from llama2.c
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_normal(self):
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))
    
    def get_logits_(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out_norm(x)
        return self.lm_head(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args :
            tokens (torch.Tensor) : input shaped (B, s, vocab_size) with s in [1; S]
        """
        x = self.embedding(tokens)
        x = self.core(x)
        logits = self.get_logits_(x)

        return logits #(B, S, vocab_size)  


class LanguageModelForSAE(LanguageModel):
    def __init__(self, model_config: TransformerConfig) -> None:
        super().__init__(model_config)

    def forward(self, tokens: torch.Tensor, act: bool = False, stop_at_layer: int = -1) -> torch.Tensor:
        """
        Args :
            - tokens (torch.Tensor) : input shaped (B, s, vocab_size) with s in [1; S]
            - act (bool) : return hidden activations if True
            - stop_at_layer (int) : the indice of the DecoderLayer module to take output
        """
        x = self.embedding(tokens)
        x = self.core(x, stop_at_layer=stop_at_layer)

        if act:
            return x #(B, S, d_model)
        
        logits = self.get_logits_(x)

        return logits #(B, S, vocab_size)


def load_transformer_model(filename: str, config: TransformerConfig, device="cpu", verbose: bool = True, class_model=LanguageModel):
    model = class_model(config)
    model.load_state_dict(
        torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
        )
    if verbose:
        print(model)
    return model.to(device)


def sample(model: LanguageModel, tokenizer: CharTokenizer, prompt: str = "", device="cpu", g = None) -> str:
    idx = torch.tensor(
        [tokenizer.char_to_int[SOS]] + tokenizer(prompt),
        dtype=torch.int32,
        device=device
        ).unsqueeze(0)
    
    next_id = -1

    while next_id != tokenizer.char_to_int[EOS]:
        logits = model(idx) # (1, len_s, d_model)

        # calcul des probas pour chaque élément du vocabulaire
        probs = F.softmax(logits[:, -1, :], dim=-1)
        # tirage au sort en prenant en compte ces probas
        next_id = torch.multinomial(probs, num_samples=1, generator=g).item()
        # concaténation
        idx = torch.cat([idx, torch.tensor(next_id, device=device).view(1, 1)], dim=1)

        if idx.shape[1] > model.config.max_len:
            break
    
    return tokenizer.to_string(idx[0].tolist())