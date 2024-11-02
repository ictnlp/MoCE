# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, AdaptiveMultiscaleAttention
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
import random
import torch.nn.functional as F
import pdb

class TransformerAdaptiveMultiscaleEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        self.print_expert_weight = getattr(args, "print_expert_weight", False)
    def build_self_attention(self, embed_dim, args):
        return AdaptiveMultiscaleAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            conv_kernels=args.conv_kernels,
            force_expert_ratio=getattr(args, "force_expert_ratio", 0.0),
            token_level=getattr(args, "token_level_adaptive", False),
            langid_expert=getattr(args, "langid_expert", False)
        )

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None, langid_embeddings=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            # need_weights=True,
            attn_mask=attn_mask,
            print_expert_weight=self.print_expert_weight,
            langid_embeddings=langid_embeddings,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x
        # return x, _

# class TransformerAdaptiveMultiscaleDecoderLayer(TransformerDecoderLayer):
#     def build_self_attention(
#         self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
#     ):
#         return AdaptiveMultiscaleAttention(
#             embed_dim,
#             args.decoder_attention_heads,
#             dropout=args.attention_dropout,
#             add_bias_kv=add_bias_kv,
#             add_zero_attn=add_zero_attn,
#             self_attention=not getattr(args, "cross_self_attention", False),
#             q_noise=self.quant_noise,
#             qn_block_size=self.quant_noise_block_size,
#             conv_kernels=args.conv_kernels,
#             left_pad=True
#         )

#     def build_encoder_attention(self, embed_dim, args):
#         return AdaptiveMultiscaleAttention(
#             embed_dim,
#             args.decoder_attention_heads,
#             kdim=getattr(args, "encoder_embed_dim", None),
#             vdim=getattr(args, "encoder_embed_dim", None),
#             dropout=args.attention_dropout,
#             encoder_decoder_attention=True,
#             q_noise=self.quant_noise,
#             qn_block_size=self.quant_noise_block_size,
#             conv_kernels=args.conv_kernels,
#             left_pad=True
#         )