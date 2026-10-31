import torch
from torch import nn
from ultralytics.nn.modules.utils import _get_clones
from .model_misc import get_valid_ratio

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer that performs self-attention followed by cross-attention.

    This layer was previously called TransformerDecoderLayer but was renamed to better reflect its role in the
    architecture. It processes input sequences through self-attention and then cross-attention with another input
    (typically image features).

    The layer supports both pre-norm and post-norm configurations, as well as positional encoding at different stages of
    the attention mechanism.
    """

    def __init__(self, d_model, dim_feedforward, dropout, pos_enc_at_attn, pos_enc_at_cross_attn_keys, pos_enc_at_cross_attn_queries, pre_norm, self_attention=None, cross_attention=None):
        """Initialize a transformer encoder layer.

        Args:
            d_model: Model dimension/hidden size
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            pos_enc_at_attn: Whether to add positional encodings at self-attention
            pos_enc_at_cross_attn_keys: Whether to add positional encodings to keys in cross-attention
            pos_enc_at_cross_attn_queries: Whether to add positional encodings to queries in cross-attention
            pre_norm: Whether to use pre-norm (True) or post-norm (False) architecture
            self_attention: Self-attention module
            cross_attention: Cross-attention module for attending to image features
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention or nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256)
        self.cross_attn_image = cross_attention or nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.layer_idx = None

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None, **kwargs):
        """Forward pass for post-norm architecture.

        In post-norm architecture, normalization is applied after attention and feedforward operations.

        Args:
            tgt (torch.Tensor): Input tensor to be processed.
            memory (torch.Tensor): Memory tensor for cross-attention.
            tgt_mask (torch.Tensor): Mask for self-attention.
            memory_mask (torch.Tensor): Mask for cross-attention.
            tgt_key_padding_mask (torch.Tensor): Key padding mask for self-attention.
            memory_key_padding_mask (torch.Tensor): Key padding mask for cross-attention.
            pos (torch.Tensor): Positional encoding for memory.
            query_pos (torch.Tensor): Positional encoding for query.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Processed tensor
        """
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn_image(query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt, key=memory + pos if self.pos_enc_at_cross_attn_keys else memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        """Forward pass for pre-norm architecture.

        In pre-norm architecture, normalization is applied before attention and feedforward operations.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor for cross-attention
            dac: Whether to use Divide-and-Conquer attention
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query

        Returns:
            Processed tensor
        """
        if dac:
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2:]
            tgt = tgt[:tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt).contiguous()
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        memory = memory.to(tgt2.dtype).contiguous()
        tgt2 = self.cross_attn_image(query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2, key=memory + pos if self.pos_enc_at_cross_attn_keys else memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, dac=False, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        """Forward pass for the transformer encoder layer.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor (e.g., image features) for cross-attention
            dac: Whether to use Divide-and-Conquer attention (only apply self-attention to first half)
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query

        Returns:
            Processed tensor after self-attention, cross-attention, and feedforward network
        """
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(tgt, memory, dac=dac, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)

class TransformerEncoder(nn.Module):
    """Transformer encoder that processes multi-level features.

    This encoder takes multi-level features (e.g., from a backbone network) and processes them through a stack of
    transformer encoder layers. It supports features from multiple levels (e.g., different resolutions) and can apply
    activation checkpointing for memory efficiency during training.

    Args:
        layer: The encoder layer to be stacked multiple times
        num_layers: Number of encoder layers to stack
        d_model: Model dimension/hidden size
        num_feature_levels: Number of feature levels to process
        frozen: Whether to freeze the parameters of this module
        use_act_checkpoint: Whether to use activation checkpointing during training
    """

    def __init__(self, layer, num_layers, d_model, num_feature_levels, frozen=False, use_act_checkpoint=False):
        """Initialize the transformer encoder."""
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)
        self.use_act_checkpoint = use_act_checkpoint
        for (layer_idx, encoder_layer) in enumerate(self.layers):
            encoder_layer.layer_idx = layer_idx

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        """Prepare multi-level features for transformer encoder."""
        assert len(srcs) == self.num_feature_levels, 'mismatch between expected and received # of feature levels'
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        has_mask = masks is not None and masks[0] is not None
        for (lvl, (src, mask, pos_embed)) in enumerate(zip(srcs, masks, pos_embeds)):
            src = src.flatten(2).transpose(1, 2)
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones((src_flatten.shape[0], self.num_feature_levels, 2), device=src_flatten.device, dtype=src_flatten.dtype)
        return (src_flatten, mask_flatten, lvl_pos_embed_flatten, valid_ratios)

    def forward(self, src, src_key_padding_masks=None, pos=None, prompt=None, prompt_key_padding_mask=None, encoder_extra_kwargs=None):
        """Process multi-level features through the transformer encoder.

        Args:
            src: List of multi-level features, each with shape (batch_size, channels, height, width)
            src_key_padding_masks: List of padding masks for each feature level, each with shape (batch_size, height,
                width)
            pos: List of positional embeddings for each feature level, each with shape (batch_size, channels, height,
                width)
            prompt: Optional text/prompt features to attend to, with shape (seq_len, batch_size, d_model)
            prompt_key_padding_mask: Optional padding mask for prompt, with shape (batch_size, seq_len)
            encoder_extra_kwargs: Optional additional arguments to pass to each encoder layer

        Returns:
            A tuple containing:
            - output: Processed features with shape (seq_len, batch_size, d_model)
            - key_padding_masks_flatten: Flattened padding masks
            - lvl_pos_embed_flatten: Flattened positional embeddings
            - valid_ratios: Valid ratios for each feature level
        """
        assert len(src) == self.num_feature_levels, 'must be equal to num_feature_levels'
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels
        (src_flatten, key_padding_masks_flatten, lvl_pos_embed_flatten, valid_ratios) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)
        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}
            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs['memory'] = prompt
            layer_kwargs['memory_key_padding_mask'] = prompt_key_padding_mask
            layer_kwargs['query_pos'] = lvl_pos_embed_flatten
            layer_kwargs['tgt'] = output
            layer_kwargs['tgt_key_padding_mask'] = key_padding_masks_flatten
            if self.training:
                assert self.use_act_checkpoint, 'activation ckpt not enabled in encoder'
            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = layer(**layer_kwargs)
        return (output.transpose(0, 1), key_padding_masks_flatten.transpose(0, 1) if key_padding_masks_flatten is not None else None, lvl_pos_embed_flatten.transpose(0, 1), valid_ratios)

class TransformerEncoderFusion(TransformerEncoder):
    """Transformer encoder that fuses text and image features.

    This encoder extends TransformerEncoder to handle both text and image features, with the ability to add pooled text
    features to image features for better cross-modal fusion. It supports torch.compile for performance optimization.

    Args:
        layer (nn.Module): The encoder layer to be stacked multiple times.
        num_layers (int): Number of encoder layers to stack.
        d_model (int): Model dimension/hidden size.
        num_feature_levels (int): Number of feature levels to process.
        add_pooled_text_to_img_feat (bool): Whether to add pooled text features to image features.
        pool_text_with_mask (bool): Whether to use the mask when pooling text features.
        compile_mode (str | None): Mode for torch.compile, or None to disable compilation.
        **kwargs (Any): Additional arguments to pass to the parent class.
    """

    def __init__(self, layer, num_layers, d_model, num_feature_levels, add_pooled_text_to_img_feat=True, pool_text_with_mask=False, compile_mode=None, **kwargs):
        """Initialize the transformer encoder with text-image fusion."""
        super().__init__(layer, num_layers, d_model, num_feature_levels, **kwargs)
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)

    def forward(self, src, prompt, src_key_padding_mask=None, src_pos=None, prompt_key_padding_mask=None, feat_sizes=None, encoder_extra_kwargs=None):
        """Forward pass for the transformer encoder with text-image fusion."""
        bs = src[0].shape[1]
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for (i, (h, w)) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1) if src_key_padding_mask[i] is not None else None
        else:
            assert all((x.dim() == 4 for x in src)), 'expected list of (bs, c, h, w) tensors'
        if self.add_pooled_text_to_img_feat:
            pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, self.pool_text_with_mask)
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
            src = [x.add_(pooled_text) for x in src]
        (out, key_padding_masks_flatten, lvl_pos_embed_flatten, valid_ratios) = super().forward(src, src_key_padding_masks=src_key_padding_mask, pos=src_pos, prompt=prompt.transpose(0, 1), prompt_key_padding_mask=prompt_key_padding_mask, encoder_extra_kwargs=encoder_extra_kwargs)
        return {'memory': out, 'padding_mask': key_padding_masks_flatten, 'pos_embed': lvl_pos_embed_flatten, 'memory_text': prompt, 'valid_ratios': valid_ratios}

def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    """Mean-pool the prompt embeddings over the valid tokens only."""
    if not pool_with_mask:
        return prompt.mean(dim=0)
    assert prompt_mask.dim() == 2
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text