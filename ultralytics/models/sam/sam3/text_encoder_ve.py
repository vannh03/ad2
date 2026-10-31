from collections import OrderedDict
from typing import Callable
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .model_misc import LayerScale

class ResidualAttentionBlock(nn.Module):
    """Transformer block with multi-head attention, layer normalization, and MLP feed-forward network."""

    def __init__(self, d_model, n_head, mlp_ratio=4.0, ls_init_value=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Initialize residual attention block with configurable dimensions and normalization."""
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, mlp_width)), ('gelu', act_layer()), ('c_proj', nn.Linear(mlp_width, d_model))]))

    def attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        """Compute multi-head attention with optional cross-attention support and masking."""
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q_x.dtype)
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
        """Apply residual attention with layer normalization and MLP, supporting optional cross-attention."""
        k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    """Stack of residual attention blocks forming a transformer encoder with optional gradient checkpointing."""

    def __init__(self, width, layers, heads, mlp_ratio=4.0, ls_init_value=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, compile_mode=None, use_act_checkpoint=False):
        """Initialize transformer with configurable depth, width, and optional compilation/checkpointing."""
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = use_act_checkpoint
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer) for _ in range(layers)])
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)
            if self.grad_checkpointing:
                torch._dynamo.config.optimize_ddp = False

    def forward(self, x, attn_mask=None):
        """Process input through all transformer blocks with optional gradient checkpointing during training."""
        for (_, r) in enumerate(self.resblocks):
            if self.grad_checkpointing and (not torch.jit.is_scripting()) and self.training:
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

def text_global_pool(x, text=None, pool_type='argmax'):
    """Extract pooled representation and tokens from text embeddings using specified pooling strategy
    (first/last/argmax/none).
    """
    if pool_type == 'first':
        (pooled, tokens) = (x[:, 0], x[:, 1:])
    elif pool_type == 'last':
        (pooled, tokens) = (x[:, -1], x[:, :-1])
    elif pool_type == 'argmax':
        assert text is not None
        (pooled, tokens) = (x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x)
    else:
        pooled = tokens = x
    return (pooled, tokens)

class TextTransformer(nn.Module):
    """Text transformer encoder with causal masking and flexible pooling strategies."""

    def __init__(self, context_length=77, vocab_size=49408, width=512, heads=8, layers=12, mlp_ratio=4.0, ls_init_value=None, output_dim=512, no_causal_mask=False, pool_type='none', proj_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm, output_tokens=False, use_ln_post=True, compile_mode=None, use_act_checkpoint=False):
        """Initialize text transformer with embedding layers, transformer blocks, and pooling options."""
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(width=width, layers=layers, heads=heads, mlp_ratio=mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer, compile_mode=compile_mode, use_act_checkpoint=use_act_checkpoint)
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self):
        """Create a causal attention mask to prevent attention to future tokens."""
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def forward(self, text):
        """Forward pass through the text transformer, returning pooled output and optionally token embeddings."""
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]
        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)
        (pooled, tokens) = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return (pooled, tokens)
        return pooled

class VETextEncoder(nn.Module):
    """Text encoder for Vision Encoder (VE) models, combining a text transformer and a linear resizer."""

    def __init__(self, d_model, tokenizer, width=1024, heads=16, layers=24, context_length=32, vocab_size=49408, use_ln_post=True, compile_mode=None, use_act_checkpoint=True):
        """Initialize VE text encoder with a text transformer and a linear resizer to match decoder dimensions."""
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer
        self.encoder = TextTransformer(context_length=self.context_length, vocab_size=vocab_size, width=width, heads=heads, layers=layers, output_tokens=True, use_ln_post=use_ln_post, compile_mode=compile_mode, use_act_checkpoint=use_act_checkpoint)
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def forward(self, text, input_boxes=None):
        """Encode text input, either raw strings or pre-encoded tensors, and resize to match decoder dimensions."""
        if isinstance(text[0], str):
            assert input_boxes is None or len(input_boxes) == 0, 'not supported'
            tokenized = self.tokenizer(text, context_length=self.context_length).to(self.resizer.weight.device)
            text_attention_mask = (tokenized != 0).bool()
            inputs_embeds = self.encoder.token_embedding(tokenized)
            (_, text_memory) = self.encoder(tokenized)
            assert text_memory.shape[1] == inputs_embeds.shape[1]
            text_attention_mask = text_attention_mask.ne(1)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = self.resizer(text_memory)
        else:
            (text_attention_mask, text_memory_resized, tokenized) = text
            inputs_embeds = tokenized['inputs_embeds']
            assert input_boxes is None or len(input_boxes) == 0, "Can't replace boxes in text if it's already encoded"
        return (text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1))