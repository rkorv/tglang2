from typing import Any, Tuple

import numpy as np
import tensorflow as tf

from utils import vocab


def _stable_softmax(logits, axis=None, name=None):
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


def shape_list(tensor):
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class TF_TGLangStructureEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.es = config.structure_embedding_size

        with tf.name_scope("naming_type_embeddings"):
            self.naming_type_embeddings = self.add_weight(
                name="naming_type_embeddings", shape=[self.config.naming_types_num, self.es]
            )

        with tf.name_scope("group_type_embeddings"):
            self.group_type_embeddings = self.add_weight(
                name="group_type_embeddings", shape=[self.config.group_types_num, self.es]
            )

        with tf.name_scope("lines_num_embeddings"):
            self.lines_num_embeddings = self.add_weight(
                name="lines_num_embeddings", shape=[self.config.max_lines_num, self.es]
            )

    def call(self, naming_types, group_types, line_ids):
        naming_type_embeddings = tf.gather(self.naming_type_embeddings, naming_types)
        group_type_embeddings = tf.gather(self.group_type_embeddings, group_types)
        line_embeddings = tf.gather(self.lines_num_embeddings, line_ids)

        return naming_type_embeddings + group_type_embeddings + line_embeddings


class TF_TGLangWordEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        with tf.name_scope("word_embeddings"):
            self.word_embeddings = self.add_weight(
                name="word_embeddings", shape=[self.config.vocab_size, self.config.embedding_size]
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="position_embeddings",
                shape=[self.config.max_input_length, self.config.embedding_size],
            )

    def call(self, input_ids, position_ids):
        inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeds = tf.gather(self.position_embeddings, position_ids)
        final_embeddings = inputs_embeds + position_embeds
        return final_embeddings


class TF_TGLang2Attention(tf.keras.layers.Layer):
    # from wav2vec2: https://huggingface.co/docs/transformers/model_doc/wav2vec2

    def __init__(self, embed_dim, num_heads, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads) == self.embed_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor, seq_len, bsz):
        return tf.transpose(
            tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )

    def call(self, hidden_states):
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        query_states = self.q_proj(hidden_states) * self.scaling

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        attn_weights = _stable_softmax(attn_weights, axis=-1)
        attn_output = tf.matmul(attn_weights, value_states)

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)),
            (0, 2, 1, 3),
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output


class TF_TGLang2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.intermediate_dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        self.intermediate_act_fn = tf.keras.activations.relu

        self.output_dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            bias_initializer="zeros",
            name="output_dense",
        )

    def call(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class TF_TGLang2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TF_TGLang2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            name="attention",
        )
        self.feed_forward = TF_TGLang2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(self, hidden_states):
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class TF_TGLang2Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layers = [TF_TGLang2EncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    def call(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)

        return hidden_states


class TF_TGLangConvBlock(tf.keras.layers.Layer):
    def __init__(self, ic, oc, k, p=1, s=1, use_ln=False, use_bn=False):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(oc, k, strides=s, padding="same", use_bias=True)
        self.act = tf.keras.layers.ReLU()

        self.use_ln = use_ln
        self.use_bn = use_bn

        if use_ln:
            self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        elif use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, name="batch_norm")

    def call(self, x):
        x = self.conv(x)
        if self.use_ln:
            x = self.ln(x)
        elif self.use_bn:
            x = self.bn(x)
        return self.act(x)


class TF_TGLangConvEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        cbs = config.conv_bottleneck_hidden
        hs = config.hidden_size
        es = config.embedding_size
        ft = config.conv_bottleneck_features

        self.word_conv = TF_TGLangConvBlock(es, ft, 3, p=2, s=1)
        self.structure_conv = TF_TGLangConvBlock(es, ft, 3, p=2, s=1)

        self.encoder = tf.keras.Sequential(
            [
                TF_TGLangConvBlock(ft, cbs[0], 7, p=3, s=2, use_bn=True),
                TF_TGLangConvBlock(cbs[0], cbs[1], 5, p=2, s=2, use_bn=True),
                TF_TGLangConvBlock(cbs[1], cbs[2], 3, p=1, s=2, use_bn=True),
            ]
        )
        self.conv_bottleneck = TF_TGLangConvBlock(ft, cbs[2], 9, p=4, s=8, use_bn=True)
        self.final_conv = TF_TGLangConvBlock(cbs[2], hs, 3, p=1, s=2, use_bn=True)

    def call(self, w_embedding, s_embedding):
        # input shape: [batch_size, seq_len, embedding_size]
        x1 = self.word_conv(w_embedding) + self.structure_conv(s_embedding)
        x2 = self.encoder(x1)
        x3 = self.conv_bottleneck(x1)
        x = self.final_conv(x2 + x3)
        return x


class TF_TGLangModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = TF_TGLangWordEmbeddings(config)
        self.structure_embeddings = TF_TGLangStructureEmbeddings(config)

        self.conv_encoder = TF_TGLangConvEncoder(config)
        self.transformer_encoder = TF_TGLang2Encoder(config)

        self.code_classifier = tf.keras.layers.Dense(config.num_classes, name="code_classifier")
        # self.other_classifier = tf.keras.layers.Dense(2, name="other_classifier")

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.int64, name="input_ids"),
            tf.TensorSpec(shape=[None], dtype=tf.int64, name="naming_types"),
            tf.TensorSpec(shape=[None], dtype=tf.int64, name="group_types"),
            tf.TensorSpec(shape=[None], dtype=tf.int64, name="line_ids"),
            tf.TensorSpec(shape=[None], dtype=tf.int64, name="position_ids"),
        ]
    )
    def call(self, input_ids, naming_types, group_types, line_ids, position_ids):
        input_ids = input_ids[None]
        naming_types = naming_types[None]
        position_ids = position_ids[None]
        group_types = group_types[None]
        line_ids = line_ids[None]

        w_embedding = self.word_embeddings(input_ids, position_ids)
        s_embedding = self.structure_embeddings(naming_types, group_types, line_ids)

        hidden_states = self.conv_encoder(w_embedding, s_embedding)

        hidden_states = self.transformer_encoder(hidden_states)
        features = hidden_states[:, 0]

        code_logits = self.code_classifier(features)[0]
        # other_logits = self.other_classifier(features)[0]

        label = tf.argmax(code_logits)
        label_conf = _stable_softmax(code_logits)[label]

        # is_code = tf.argmax(other_logits)
        # is_code_conf = _stable_softmax(other_logits)[1]

        return label, label_conf
