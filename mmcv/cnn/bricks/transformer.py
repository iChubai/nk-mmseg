"""Transformer bricks implemented with Jittor-compatible operators."""

from __future__ import annotations

from typing import Optional

import jittor as jt
import jittor.nn as nn

from .drop import build_dropout


def _build_activation(act_cfg):
    if act_cfg is None:
        return nn.Identity()
    cfg = dict(act_cfg) if isinstance(act_cfg, dict) else {'type': act_cfg}
    act_type = str(cfg.pop('type', 'ReLU'))
    t = act_type.lower()
    if t == 'relu':
        return nn.ReLU()
    if t == 'gelu':
        return nn.GELU()
    if t in ('silu', 'swish'):
        if hasattr(nn, 'SiLU'):
            return nn.SiLU()
        return nn.ReLU()
    if t == 'quickgelu':
        try:
            from mmseg.registry import MODELS
            return MODELS.build(dict(type='QuickGELU'))
        except Exception:
            return nn.GELU()
    # Try registry-defined custom activations.
    try:
        from mmseg.registry import MODELS
        return MODELS.build(dict(type=act_type, **cfg))
    except Exception:
        pass
    return nn.ReLU()


class _MultiheadAttentionCore(nn.Module):
    """A lightweight replacement for torch.nn.MultiheadAttention."""

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, bias=True):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})'
            )
        self.embed_dims = int(embed_dims)
        self.embed_dim = self.embed_dims
        self.num_heads = int(num_heads)
        self.head_dims = self.embed_dims // self.num_heads
        self.scale = self.head_dims**-0.5
        self.dropout = float(attn_drop)
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False

        self.q_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=bias)
        self.k_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=bias)
        self.v_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=bias)
        self.out_proj = nn.Linear(self.embed_dims, self.embed_dims, bias=bias)
        self.attn_drop = nn.Dropout(float(attn_drop))

    @property
    def in_proj_weight(self):
        return jt.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)

    @property
    def in_proj_bias(self):
        q_bias = getattr(self.q_proj, 'bias', None)
        k_bias = getattr(self.k_proj, 'bias', None)
        v_bias = getattr(self.v_proj, 'bias', None)
        if q_bias is None or k_bias is None or v_bias is None:
            return None
        return jt.concat([q_bias, k_bias, v_bias], dim=0)

    def _split_heads(self, x):
        # x: (seq_len, batch, embed) -> (batch, heads, seq_len, head_dim)
        seq_len, batch, _ = x.shape
        x = x.permute((1, 0, 2))
        x = x.reshape((batch, seq_len, self.num_heads, self.head_dims))
        return x.permute((0, 2, 1, 3))

    def _merge_heads(self, x):
        # x: (batch, heads, seq_len, head_dim) -> (seq_len, batch, embed)
        batch, _, seq_len, _ = x.shape
        x = x.permute((0, 2, 1, 3)).reshape((batch, seq_len, self.embed_dims))
        return x.permute((1, 0, 2))

    def _apply_masks(self, scores, attn_mask=None, key_padding_mask=None):
        # scores: (batch, heads, q_len, k_len)
        if attn_mask is not None:
            if not isinstance(attn_mask, jt.Var):
                attn_mask = jt.array(attn_mask)
            if attn_mask.ndim == 2:
                # (q_len, k_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                # (batch, q_len, k_len) or (batch*heads, q_len, k_len)
                if attn_mask.shape[0] == scores.shape[0]:
                    attn_mask = attn_mask.unsqueeze(1)
                elif attn_mask.shape[0] == scores.shape[0] * scores.shape[1]:
                    attn_mask = attn_mask.reshape((scores.shape[0],
                                                   scores.shape[1],
                                                   scores.shape[2],
                                                   scores.shape[3]))
                else:
                    attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.ndim != 4:
                raise ValueError(
                    f'Unsupported attn_mask ndim={attn_mask.ndim}, expected 2/3/4'
                )

            if attn_mask.dtype == jt.bool:
                neg = jt.array(-1e4, dtype=scores.dtype)
                scores = jt.where(attn_mask, neg, scores)
            else:
                scores = scores + attn_mask.astype(scores.dtype)

        if key_padding_mask is not None:
            if not isinstance(key_padding_mask, jt.Var):
                key_padding_mask = jt.array(key_padding_mask)
            if key_padding_mask.ndim != 2:
                raise ValueError(
                    f'key_padding_mask should be 2D, got ndim={key_padding_mask.ndim}'
                )
            mask = key_padding_mask.astype(jt.bool).unsqueeze(1).unsqueeze(1)
            neg = jt.array(-1e4, dtype=scores.dtype)
            scores = jt.where(mask, neg, scores)
        return scores

    def execute(self,
                query,
                key,
                value,
                key_padding_mask=None,
                need_weights=True,
                attn_mask=None):
        # query/key/value: (seq_len, batch, embed_dims)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = jt.matmul(q, k.permute((0, 1, 3, 2))) * self.scale
        scores = self._apply_masks(
            scores, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        attn = nn.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = jt.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        if need_weights:
            # Match PyTorch default: average over heads -> (batch, q_len, k_len)
            weights = attn.mean(dim=1)
            return out, weights
        return out, None


class BaseTransformerLayer(nn.Module):

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=None,
                 operation_order=None,
                 norm_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__()
        del kwargs
        self.batch_first = bool(batch_first)
        self.operation_order = tuple(operation_order or ('self_attn', 'norm',
                                                         'ffn', 'norm'))

        self.attentions = nn.ModuleList()
        embed_dims = None
        if attn_cfgs is not None:
            if isinstance(attn_cfgs, dict):
                attn_cfgs = [attn_cfgs]
            for cfg in attn_cfgs:
                cfg = dict(cfg)
                layer_type = cfg.pop('type', 'MultiheadAttention')
                cfg.setdefault('batch_first', self.batch_first)
                if layer_type in ('MultiheadAttention', MultiheadAttention):
                    self.attentions.append(MultiheadAttention(**cfg))
                else:
                    self.attentions.append(
                        build_transformer_layer(dict(type=layer_type, **cfg)))
                if embed_dims is None and 'embed_dims' in cfg:
                    embed_dims = int(cfg['embed_dims'])

        self.ffns = nn.ModuleList()
        if ffn_cfgs is not None:
            if isinstance(ffn_cfgs, dict):
                ffn_cfgs = [ffn_cfgs]
            for cfg in ffn_cfgs:
                cfg = dict(cfg)
                layer_type = cfg.pop('type', 'FFN')
                if layer_type in ('FFN', FFN):
                    self.ffns.append(FFN(**cfg))
                else:
                    self.ffns.append(
                        build_transformer_layer(dict(type=layer_type, **cfg)))
                if embed_dims is None and 'embed_dims' in cfg:
                    embed_dims = int(cfg['embed_dims'])
        if embed_dims is None:
            embed_dims = 256

        self.norms = nn.ModuleList()
        norm_cfg = dict(norm_cfg or {'type': 'LN'})
        norm_type = str(norm_cfg.get('type', 'LN'))
        eps = float(norm_cfg.get('eps', 1e-5))
        num_norms = sum(1 for op in self.operation_order if op == 'norm')
        for _ in range(num_norms):
            if norm_type in ('LN', 'LayerNorm'):
                self.norms.append(nn.LayerNorm(embed_dims, eps=eps))
            else:
                # Token sequence inputs are best normalized with LN in this
                # compatibility layer.
                self.norms.append(nn.LayerNorm(embed_dims, eps=eps))

    @staticmethod
    def _format_attn_masks(attn_masks, attn_mask, num_attn):
        masks = attn_masks if attn_masks is not None else attn_mask
        if num_attn <= 0:
            return []
        if masks is None:
            return [None] * num_attn
        if isinstance(masks, (list, tuple)):
            masks = list(masks)
            if len(masks) == 1 and num_attn > 1:
                masks = masks * num_attn
            if len(masks) < num_attn:
                masks = masks + [None] * (num_attn - len(masks))
            return masks[:num_attn]
        return [masks] * num_attn

    def execute(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        del kwargs
        x = query
        if key is None:
            key = x
        if value is None:
            value = key

        attn_i = 0
        norm_i = 0
        ffn_i = 0
        masks = self._format_attn_masks(attn_masks, attn_mask,
                                        len(self.attentions))

        for layer in self.operation_order:
            if layer == 'self_attn':
                x = self.attentions[attn_i](
                    query=x,
                    key=x,
                    value=x,
                    identity=x,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=masks[attn_i] if attn_i < len(masks) else None,
                    key_padding_mask=query_key_padding_mask)
                attn_i += 1
            elif layer == 'cross_attn':
                x = self.attentions[attn_i](
                    query=x,
                    key=key,
                    value=value,
                    identity=x,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=masks[attn_i] if attn_i < len(masks) else None,
                    key_padding_mask=key_padding_mask)
                attn_i += 1
            elif layer == 'norm':
                x = self.norms[norm_i](x)
                norm_i += 1
            elif layer == 'ffn':
                x = self.ffns[ffn_i](x, identity=x)
                ffn_i += 1
            else:
                raise ValueError(
                    f'Unsupported operation "{layer}" in BaseTransformerLayer')
        return x


class MultiheadAttention(nn.Module):
    """MMCV-style attention wrapper."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 dropout_layer=None,
                 batch_first=False,
                 bias=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        del init_cfg, kwargs
        self.batch_first = bool(batch_first)
        self.attn = _MultiheadAttentionCore(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop,
            bias=bias)
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def _forward_impl(self,
                      query,
                      key=None,
                      value=None,
                      identity=None,
                      query_pos=None,
                      key_pos=None,
                      attn_mask=None,
                      key_padding_mask=None,
                      need_weights=False,
                      **kwargs):
        del kwargs
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, weights = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask)

        if self.batch_first:
            out = out.transpose(0, 1)

        out = self.proj_drop(out)
        out = identity + self.dropout_layer(out)
        if need_weights:
            return out, weights
        return out

    def forward(self, *args, **kwargs):
        return self._forward_impl(*args, **kwargs)

    def execute(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
                **kwargs):
        # Some upstream modules subclass this class and implement torch-style
        # `forward(x, hw_shape, identity=...)`. Respect that override.
        if type(self) is not MultiheadAttention and type(
                self).forward is not MultiheadAttention.forward:
            return self.forward(query, key, identity=identity)
        return self._forward_impl(
            query=query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            **kwargs)


class FFN(nn.Module):
    """MMCV-style feed-forward network block."""

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU'),
                 ffn_drop=0.0,
                 dropout=0.0,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        del init_cfg, kwargs
        if int(num_fcs) < 2:
            raise ValueError(f'num_fcs must be >=2, got {num_fcs}')

        self.embed_dims = int(embed_dims)
        hidden_dims = int(feedforward_channels)
        num_fcs = int(num_fcs)
        self.add_identity = bool(add_identity)
        drop_p = float(ffn_drop if ffn_drop is not None else dropout)
        if drop_p <= 0.0 and dropout and ffn_drop == 0.0:
            drop_p = float(dropout)

        layers = []
        in_dims = self.embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_dims, hidden_dims))
            layers.append(_build_activation(act_cfg))
            layers.append(nn.Dropout(drop_p))
            in_dims = hidden_dims
        layers.append(nn.Linear(in_dims, self.embed_dims))
        layers.append(nn.Dropout(drop_p))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def execute(self, x, identity: Optional[jt.Var] = None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


def build_transformer_layer(cfg, *args, **kwargs):
    if cfg is None:
        return nn.Identity()
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg should be dict, but got {type(cfg)}')
    cfg = dict(cfg)
    layer_type = cfg.pop('type', None)

    if layer_type in ('BaseTransformerLayer', BaseTransformerLayer):
        return BaseTransformerLayer(*args, **cfg, **kwargs)
    if layer_type in ('MultiheadAttention', MultiheadAttention):
        return MultiheadAttention(*args, **cfg, **kwargs)
    if layer_type in ('FFN', FFN):
        return FFN(*args, **cfg, **kwargs)

    if isinstance(layer_type, str) and layer_type in ('DynamicConv',
                                                      'KernelUpdator'):
        try:
            from mmseg.models.decode_heads.knet_head import KernelUpdator
            return KernelUpdator(*args, **cfg, **kwargs)
        except Exception:
            return nn.Identity()

    try:
        from mmseg.registry import MODELS
        return MODELS.build(dict(type=layer_type, **cfg), **kwargs)
    except Exception:
        return nn.Identity()


__all__ = [
    'BaseTransformerLayer',
    'MultiheadAttention',
    'FFN',
    'build_dropout',
    'build_transformer_layer',
]
