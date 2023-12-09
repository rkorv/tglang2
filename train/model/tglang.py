import math

from dataclasses import make_dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils import vocab, lang_enum


class TGLangStructureEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.es = config.structure_embedding_size

        self.naming_type_embeddings = nn.Embedding(config.naming_types_num, self.es)
        self.group_type_embeddings = nn.Embedding(config.group_types_num, self.es)
        self.lines_num_embeddings = nn.Embedding(config.max_lines_num, self.es)

    def forward(self, naming_types=None, group_types=None, line_ids=None):
        naming_type_embeddings = self.naming_type_embeddings(naming_types)
        group_type_embeddings = self.group_type_embeddings(group_types)
        lines_num_embeddings = self.lines_num_embeddings(line_ids)
        embeddings = naming_type_embeddings + group_type_embeddings + lines_num_embeddings
        return embeddings


class TGLangWordEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.es = config.embedding_size

        self.word_embeddings = nn.Embedding(config.vocab_size, self.es)
        self.position_embeddings = nn.Embedding(config.max_input_length, self.es)

    def forward(self, input_ids=None, position_ids=None):
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class TGLangAttention(nn.Module):
    # from wav2vec2: https://huggingface.co/docs/transformers/model_doc/wav2vec2

    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads) == self.embed_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)

        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class TGLangFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class TGLangEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TGLangAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads)
        self.feed_forward = TGLangFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask=None):
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = attn_residual + hidden_states

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class TGLangTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TGLangEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])

            inverse_mask = ~expand_attention_mask
            hidden_states = hidden_states.masked_fill(inverse_mask, 0)

            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class Conv1DSame(torch.nn.Conv1d):
    """Workaround for same padding and stride>1 in Conv1d to match TF implementation"""

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        iw = x.size()[-1]

        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])

        if pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2])

        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class TGLangConvBlock(nn.Module):
    def __init__(self, ic, oc, k, s=1, p=0, use_ln=False, use_bn=False):
        super().__init__()
        self.conv = Conv1DSame(ic, oc, k, stride=s, bias=True)
        self.act = torch.nn.ReLU()

        self.use_ln = use_ln
        self.use_bn = use_bn
        if use_ln:
            self.ln = torch.nn.LayerNorm(oc, eps=1e-5)
        if use_bn:
            self.bn = torch.nn.BatchNorm1d(oc, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        if self.use_ln:
            x = self.ln(x.transpose(-2, -1)).transpose(-2, -1)
        elif self.use_bn:
            x = self.bn(x)
        return self.act(x)


class TGLangConvEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        cbs = config.conv_bottleneck_hidden
        hs = config.hidden_size
        es = config.embedding_size
        ses = config.structure_embedding_size
        ft = config.conv_bottleneck_features

        self.word_conv = TGLangConvBlock(es, ft, 3, p=2, s=1)
        self.structure_conv = TGLangConvBlock(ses, ft, 3, p=2, s=1)
        self.encoder = torch.nn.Sequential(
            TGLangConvBlock(ft, cbs[0], 7, p=3, s=2, use_bn=True),
            TGLangConvBlock(cbs[0], cbs[1], 5, p=2, s=2, use_bn=True),
            TGLangConvBlock(cbs[1], cbs[2], 3, p=1, s=2, use_bn=True),
        )
        self.conv_bottleneck = TGLangConvBlock(ft, cbs[2], 9, p=4, s=8, use_bn=True)
        self.final_conv = TGLangConvBlock(cbs[2], hs, 3, p=1, s=2, use_bn=True)

    def forward(self, w_embedding, s_embedding):
        we = w_embedding.transpose(1, 2)
        se = s_embedding.transpose(1, 2)

        x1 = self.word_conv(we) + self.structure_conv(se)
        x2 = self.encoder(x1)
        x3 = self.conv_bottleneck(x1)
        x = self.final_conv(x2 + x3)
        return x.transpose(1, 2)


class TGLangModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embeddings = TGLangWordEmbeddings(config)
        self.structure_embeddings = TGLangStructureEmbeddings(config)
        self.conv_encoder = TGLangConvEncoder(config)
        self.transformer_encoder = TGLangTransformerEncoder(config)

        self.code_classifier = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.other_classifier = torch.nn.Linear(config.hidden_size, 2)
        self.additional_classifiers = nn.ModuleList(
            [torch.nn.Linear(config.hidden_size, nc) for nc in config.additional_classifiers]
        )

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids,
        naming_types,
        group_types,
        line_ids,
        position_ids,
        attention_mask=None,
        with_additional_logits=True,
    ):
        w_embedding = self.word_embeddings(input_ids, position_ids)
        s_embedding = self.structure_embeddings(naming_types, group_types, line_ids)

        hidden_states = self.conv_encoder(w_embedding, s_embedding)

        if attention_mask is not None:
            attention_mask = (
                torch.nn.functional.interpolate(
                    attention_mask.float().unsqueeze(1),
                    size=hidden_states.size(1),
                    mode="nearest",
                ).squeeze(1)
                > 0.1
            )

        hidden_states = self.transformer_encoder(hidden_states, attention_mask)
        features = hidden_states[:, 0]

        code_logits = self.code_classifier(features)
        other_logits = self.other_classifier(features)

        if with_additional_logits:
            additional_logits = []
            for classifier in self.additional_classifiers:
                additional_logits.append(classifier(features))

            return code_logits, other_logits, features, additional_logits
        else:
            return code_logits, other_logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)


def _stable_softmax(logits, dim=None):
    return torch.softmax(logits=logits + 1e-9, dim=dim)


class TGLangInferenceModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, naming_types, group_types, line_ids, position_ids):
        input_ids = input_ids.unsqueeze(0)
        naming_types = naming_types.unsqueeze(0)
        positions = positions.unsqueeze(0)
        line_ids = line_ids.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)
        inp = (input_ids, naming_types, group_types, line_ids, position_ids)
        code_logits, other_logits = self.model(*inp, with_additional_logits=False)

        code_logits = code_logits[0]
        other_logits = other_logits[0]

        label = torch.argmax(code_logits)
        label_conf = _stable_softmax(code_logits, dim=-1)[label]

        is_code = torch.argmax(other_logits)
        is_code_conf = _stable_softmax(other_logits, dim=-1)[1]
        return label, label_conf, is_code, is_code_conf


def get_config(size="l"):
    config = {
        "vocab_size": len(vocab.vocab_list),
        "naming_types_num": 3,
        "group_types_num": 5,
        "max_lines_num": 256,
        "max_input_length": 4096,
        "num_classes": len(lang_enum.languages),
        "additional_classifiers": [2, 2],
    }
    if size == "l":
        config.update(
            {
                "embedding_size": 32,
                "structure_embedding_size": 8,
                "hidden_size": 64,
                "conv_bottleneck_hidden": [64, 96, 128],
                "conv_bottleneck_features": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
            }
        )
    else:
        raise NotImplementedError

    return make_dataclass("Config", fields=config.keys())(**config)


def get_model(size="l"):
    return TGLangModel(get_config(size=size))
