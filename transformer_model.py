#pip install -q tfds-nightly

# Pin matplotlib version to 3.2.2 since in the latest version
# transformer.ipynb fails with the following error:
# https://stackoverflow.com/questions/62953704/valueerror-the-number-of-fixedlocator-locations-5-usually-from-a-call-to-set
#pip install matplotlib==3.2.2

"""
WARNING: You are using pip version 20.2.2; however, version 20.2.3 is available.
You should consider upgrading via the '/tmpfs/src/tf_docs_env/bin/python -m pip install --upgrade pip' command.
Requirement already satisfied: matplotlib==3.2.2 in /home/kbuilder/.local/lib/python3.6/site-packages (3.2.2)
Requirement already satisfied: cycler>=0.10 in /home/kbuilder/.local/lib/python3.6/site-packages (from matplotlib==3.2.2) (0.10.0)
Requirement already satisfied: python-dateutil>=2.1 in /home/kbuilder/.local/lib/python3.6/site-packages (from matplotlib==3.2.2) (2.8.1)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /home/kbuilder/.local/lib/python3.6/site-packages (from matplotlib==3.2.2) (2.4.7)
Requirement already satisfied: kiwisolver>=1.0.1 in /home/kbuilder/.local/lib/python3.6/site-packages (from matplotlib==3.2.2) (1.2.0)
Requirement already satisfied: numpy>=1.11 in /tmpfs/src/tf_docs_env/lib/python3.6/site-packages (from matplotlib==3.2.2) (1.18.5)
Requirement already satisfied: six in /home/kbuilder/.local/lib/python3.6/site-packages (from cycler>=0.10->matplotlib==3.2.2) (1.15.0)
WARNING: You are using pip version 20.2.2; however, version 20.2.3 is available.
You should consider upgrading via the '/tmpfs/src/tf_docs_env/bin/python -m pip install --upgrade pip' command.
"""

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

#Setup input pipeline

#Use TFDS to load the Portugese-English translation dataset from the TED Talks Open Translation Project.

#This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

"""
Downloading and preparing dataset ted_hrlr_translate/pt_to_en/1.0.0 (download: 124.94 MiB, generated: Unknown size, total: 124.94 MiB) to /home/kbuilder/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0...
Shuffling and writing examples to /home/kbuilder/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0.incompleteUXSB4Q/ted_hrlr_translate-train.tfrecord
Shuffling and writing examples to /home/kbuilder/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0.incompleteUXSB4Q/ted_hrlr_translate-validation.tfrecord
Shuffling and writing examples to /home/kbuilder/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0.incompleteUXSB4Q/ted_hrlr_translate-test.tfrecord
Dataset ted_hrlr_translate downloaded and prepared to /home/kbuilder/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0. Subsequent calls will reuse this data.
"""

#Create a custom subwords tokenizer from the training dataset.

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

#Tokenized string is [7915, 1248, 7946, 7194, 13, 2799, 7877]
#The original string: Transformer is awesome.

#The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary.

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

#7915 - ---> T
#1248 - ---> ran
#7946 - ---> s
#7194 - ---> former
#13 - ---> is
#2799 - ---> awesome
#7877 - --->.

BUFFER_SIZE = 20000
BATCH_SIZE = 64

#Add a start and end token to the input and target.


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2


#You want to use Dataset.map to apply this function to each element of the dataset.Dataset.map runs in graph mode.

#Graph tensors do not have a value.
#In graph mode you can only use TensorFlow Ops and functions.
#So you can't .map this function directly: You need to wrap it in a tf.py_function. The tf.py_function will pass regular tensors (with a value and a .numpy() method to access it), to the wrapped python function.


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


#Note: To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
MAX_LENGTH = 40


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch, en_batch = next(iter(val_dataset))
pt_batch, en_batch

"""
(<tf.Tensor: shape=(64, 38), dtype=int64, numpy=
 array([[8214,  342, 3032, ...,    0,    0,    0],
        [8214,   95,  198, ...,    0,    0,    0],
        [8214, 4479, 7990, ...,    0,    0,    0],
        ...,
        [8214,  584,   12, ...,    0,    0,    0],
        [8214,   59, 1548, ...,    0,    0,    0],
        [8214,  118,   34, ...,    0,    0,    0]])>,
 <tf.Tensor: shape=(64, 40), dtype=int64, numpy=
 array([[8087,   98,   25, ...,    0,    0,    0],
        [8087,   12,   20, ...,    0,    0,    0],
        [8087,   12, 5453, ...,    0,    0,    0],
        ...,
        [8087,   18, 2059, ...,    0,    0,    0],
        [8087,   16, 1436, ...,    0,    0,    0],
        [8087,   15,   57, ...,    0,    0,    0]])>)
"""

# Positional encoding

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()

#(1, 50, 512)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)
"""
< tf.Tensor: shape = (3, 1, 1, 5), dtype = float32, numpy =
array([[[[0., 0., 1., 1., 0.]]],

       [[[0., 0., 0., 1., 1.]]],

       [[[1., 1., 1., 0., 0.]]]], dtype=float32) >
"""


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])

"""
< tf.Tensor: shape = (3, 3), dtype = float32, numpy =
array([[0., 1., 1.],
       [0., 0., 1.],
       [0., 0., 0.]], dtype=float32) >
       """


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

np.set_printoptions(suppress=True)

temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)

# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

#Attention weights are:
#tf.Tensor([[0. 1. 0. 0.]], shape=(1, 4), dtype=float32)
#Output is:
#tf.Tensor([[10.  0.]], shape=(1, 2), dtype=float32)

# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

#Attention weights are:
#tf.Tensor([[0.  0.  0.5 0.5]], shape=(1, 4), dtype=float32)
#Output is:
#tf.Tensor([[550.    5.5]], shape=(1, 2), dtype=float32)


# This query aligns equally with the first and second key,
# so their values get averaged.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

#Attention weights are:
#tf.Tensor([[0.5 0.5 0.  0. ]], shape=(1, 4), dtype=float32)
#Output is:
#tf.Tensor([[5.5 0. ]], shape=(1, 2), dtype=float32)

temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)

#Attention weights are:
#tf.Tensor(
#[[0.  0.  0.5 0.5]
# [0.  1.  0.  0. ]
# [0.5 0.5 0.  0. ]], shape=(3, 4), dtype=float32)
#Output is:
#tf.Tensor(
#[[550.    5.5]
# [ 10.    0. ]
# [  5.5   0. ]], shape=(3, 2), dtype=float32)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

#Create a MultiHeadAttention layer to try out.At each location in the sequence, y, the MultiHeadAttention runs all 8 attention heads across all other locations in the sequence, returning a new vector of the same length at each location.

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)

#(TensorShape([1, 60, 512]), TensorShape([1, 8, 60, 60]))

#Point wise feed forward network

# Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape

#TensorShape([64, 50, 512])

#Encoder and decoder

#Encoder layer

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


sample_encoder_layer = EncoderLayer(512, 8, 2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)), False, None)

sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)

#TensorShape([64, 43, 512])

#Decoder layer

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


sample_decoder_layer = DecoderLayer(512, 8, 2048)

sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
    False, None, None)

sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)

#TensorShape([64, 50, 512])

#Encoder

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

#(64, 62, 512)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(temp_input,
                              enc_output=sample_encoder_output,
                              training=False,
                              look_ahead_mask=None,
                              padding_mask=None)

output.shape, attn['decoder_layer2_block2'].shape

#(TensorShape([64, 26, 512]), TensorShape([64, 8, 26, 62]))

#Create the Transformer

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, target_vocab_size=8000,
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                               enc_padding_mask=None,
                               look_ahead_mask=None,
                               dec_padding_mask=None)

fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)

#TensorShape([64, 36, 8000])

#Set hyperparameters

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1

#Optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

#Text(0.5, 0, 'Train Step')

#Loss and metrics

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

#Training and checkpointing

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

#Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every n epochs.

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 20

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

#Portuguese is used as the input language and English is the target language.

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

"""
Epoch 1 Batch 0 Loss 8.9897 Accuracy 0.0000
Epoch 1 Batch 50 Loss 8.9449 Accuracy 0.0076
Epoch 1 Batch 100 Loss 8.8546 Accuracy 0.0171
Epoch 1 Batch 150 Loss 8.7529 Accuracy 0.0204
Epoch 1 Batch 200 Loss 8.6280 Accuracy 0.0221
Epoch 1 Batch 250 Loss 8.4755 Accuracy 0.0230
Epoch 1 Batch 300 Loss 8.3013 Accuracy 0.0237
Epoch 1 Batch 350 Loss 8.1149 Accuracy 0.0264
Epoch 1 Batch 400 Loss 7.9342 Accuracy 0.0303
Epoch 1 Batch 450 Loss 7.7682 Accuracy 0.0339
Epoch 1 Batch 500 Loss 7.6230 Accuracy 0.0370
Epoch 1 Batch 550 Loss 7.4890 Accuracy 0.0405
Epoch 1 Batch 600 Loss 7.3663 Accuracy 0.0439
Epoch 1 Batch 650 Loss 7.2483 Accuracy 0.0473
Epoch 1 Batch 700 Loss 7.1355 Accuracy 0.0506
Epoch 1 Loss 7.1309 Accuracy 0.0508
Time taken for 1 epoch: 54.13676071166992 secs

Epoch 2 Batch 0 Loss 5.5349 Accuracy 0.1011
Epoch 2 Batch 50 Loss 5.5150 Accuracy 0.1024
Epoch 2 Batch 100 Loss 5.4537 Accuracy 0.1040
Epoch 2 Batch 150 Loss 5.3991 Accuracy 0.1061
Epoch 2 Batch 200 Loss 5.3503 Accuracy 0.1081
Epoch 2 Batch 250 Loss 5.3032 Accuracy 0.1102
Epoch 2 Batch 300 Loss 5.2637 Accuracy 0.1117
Epoch 2 Batch 350 Loss 5.2216 Accuracy 0.1138
Epoch 2 Batch 400 Loss 5.1849 Accuracy 0.1155
Epoch 2 Batch 450 Loss 5.1530 Accuracy 0.1172
Epoch 2 Batch 500 Loss 5.1228 Accuracy 0.1186
Epoch 2 Batch 550 Loss 5.0952 Accuracy 0.1201
Epoch 2 Batch 600 Loss 5.0698 Accuracy 0.1214
Epoch 2 Batch 650 Loss 5.0435 Accuracy 0.1226
Epoch 2 Batch 700 Loss 5.0178 Accuracy 0.1237
Epoch 2 Loss 5.0172 Accuracy 0.1237
Time taken for 1 epoch: 29.873267650604248 secs

Epoch 3 Batch 0 Loss 4.6709 Accuracy 0.1509
Epoch 3 Batch 50 Loss 4.6441 Accuracy 0.1414
Epoch 3 Batch 100 Loss 4.6059 Accuracy 0.1434
Epoch 3 Batch 150 Loss 4.5951 Accuracy 0.1438
Epoch 3 Batch 200 Loss 4.5741 Accuracy 0.1451
Epoch 3 Batch 250 Loss 4.5602 Accuracy 0.1455
Epoch 3 Batch 300 Loss 4.5514 Accuracy 0.1459
Epoch 3 Batch 350 Loss 4.5379 Accuracy 0.1466
Epoch 3 Batch 400 Loss 4.5248 Accuracy 0.1472
Epoch 3 Batch 450 Loss 4.5114 Accuracy 0.1478
Epoch 3 Batch 500 Loss 4.5014 Accuracy 0.1483
Epoch 3 Batch 550 Loss 4.4884 Accuracy 0.1487
Epoch 3 Batch 600 Loss 4.4771 Accuracy 0.1494
Epoch 3 Batch 650 Loss 4.4628 Accuracy 0.1502
Epoch 3 Batch 700 Loss 4.4488 Accuracy 0.1510
Epoch 3 Loss 4.4481 Accuracy 0.1511
Time taken for 1 epoch: 29.877694368362427 secs

Epoch 4 Batch 0 Loss 4.0926 Accuracy 0.1622
Epoch 4 Batch 50 Loss 4.1464 Accuracy 0.1656
Epoch 4 Batch 100 Loss 4.1350 Accuracy 0.1667
Epoch 4 Batch 150 Loss 4.1380 Accuracy 0.1665
Epoch 4 Batch 200 Loss 4.1369 Accuracy 0.1664
Epoch 4 Batch 250 Loss 4.1211 Accuracy 0.1671
Epoch 4 Batch 300 Loss 4.1053 Accuracy 0.1679
Epoch 4 Batch 350 Loss 4.0913 Accuracy 0.1687
Epoch 4 Batch 400 Loss 4.0771 Accuracy 0.1696
Epoch 4 Batch 450 Loss 4.0622 Accuracy 0.1704
Epoch 4 Batch 500 Loss 4.0476 Accuracy 0.1716
Epoch 4 Batch 550 Loss 4.0322 Accuracy 0.1728
Epoch 4 Batch 600 Loss 4.0175 Accuracy 0.1740
Epoch 4 Batch 650 Loss 4.0015 Accuracy 0.1749
Epoch 4 Batch 700 Loss 3.9851 Accuracy 0.1761
Epoch 4 Loss 3.9845 Accuracy 0.1761
Time taken for 1 epoch: 29.84564232826233 secs

Epoch 5 Batch 0 Loss 3.7526 Accuracy 0.1907
Epoch 5 Batch 50 Loss 3.6250 Accuracy 0.1949
Epoch 5 Batch 100 Loss 3.6295 Accuracy 0.1947
Epoch 5 Batch 150 Loss 3.6215 Accuracy 0.1948
Epoch 5 Batch 200 Loss 3.6140 Accuracy 0.1961
Epoch 5 Batch 250 Loss 3.5995 Accuracy 0.1977
Epoch 5 Batch 300 Loss 3.5916 Accuracy 0.1985
Epoch 5 Batch 350 Loss 3.5803 Accuracy 0.1992
Epoch 5 Batch 400 Loss 3.5691 Accuracy 0.2000
Epoch 5 Batch 450 Loss 3.5618 Accuracy 0.2006
Epoch 5 Batch 500 Loss 3.5502 Accuracy 0.2013
Epoch 5 Batch 550 Loss 3.5413 Accuracy 0.2018
Epoch 5 Batch 600 Loss 3.5299 Accuracy 0.2023
Epoch 5 Batch 650 Loss 3.5208 Accuracy 0.2029
Epoch 5 Batch 700 Loss 3.5097 Accuracy 0.2034
Saving checkpoint for epoch 5 at ./checkpoints/train/ckpt-1
Epoch 5 Loss 3.5092 Accuracy 0.2035
Time taken for 1 epoch: 30.028881549835205 secs

Epoch 6 Batch 0 Loss 3.3597 Accuracy 0.2123
Epoch 6 Batch 50 Loss 3.1817 Accuracy 0.2194
Epoch 6 Batch 100 Loss 3.1879 Accuracy 0.2189
Epoch 6 Batch 150 Loss 3.1915 Accuracy 0.2196
Epoch 6 Batch 200 Loss 3.1908 Accuracy 0.2191
Epoch 6 Batch 250 Loss 3.1842 Accuracy 0.2195
Epoch 6 Batch 300 Loss 3.1819 Accuracy 0.2204
Epoch 6 Batch 350 Loss 3.1748 Accuracy 0.2201
Epoch 6 Batch 400 Loss 3.1663 Accuracy 0.2206
Epoch 6 Batch 450 Loss 3.1587 Accuracy 0.2211
Epoch 6 Batch 500 Loss 3.1505 Accuracy 0.2221
Epoch 6 Batch 550 Loss 3.1424 Accuracy 0.2225
Epoch 6 Batch 600 Loss 3.1357 Accuracy 0.2228
Epoch 6 Batch 650 Loss 3.1287 Accuracy 0.2234
Epoch 6 Batch 700 Loss 3.1213 Accuracy 0.2239
Epoch 6 Loss 3.1208 Accuracy 0.2239
Time taken for 1 epoch: 29.813305616378784 secs

Epoch 7 Batch 0 Loss 2.8050 Accuracy 0.2578
Epoch 7 Batch 50 Loss 2.7950 Accuracy 0.2410
Epoch 7 Batch 100 Loss 2.7892 Accuracy 0.2425
Epoch 7 Batch 150 Loss 2.7799 Accuracy 0.2424
Epoch 7 Batch 200 Loss 2.7793 Accuracy 0.2420
Epoch 7 Batch 250 Loss 2.7732 Accuracy 0.2415
Epoch 7 Batch 300 Loss 2.7710 Accuracy 0.2414
Epoch 7 Batch 350 Loss 2.7664 Accuracy 0.2417
Epoch 7 Batch 400 Loss 2.7551 Accuracy 0.2427
Epoch 7 Batch 450 Loss 2.7523 Accuracy 0.2432
Epoch 7 Batch 500 Loss 2.7457 Accuracy 0.2437
Epoch 7 Batch 550 Loss 2.7402 Accuracy 0.2443
Epoch 7 Batch 600 Loss 2.7351 Accuracy 0.2448
Epoch 7 Batch 650 Loss 2.7306 Accuracy 0.2453
Epoch 7 Batch 700 Loss 2.7270 Accuracy 0.2457
Epoch 7 Loss 2.7269 Accuracy 0.2457
Time taken for 1 epoch: 30.495452165603638 secs

Epoch 8 Batch 0 Loss 2.4499 Accuracy 0.2589
Epoch 8 Batch 50 Loss 2.4221 Accuracy 0.2630
Epoch 8 Batch 100 Loss 2.4297 Accuracy 0.2617
Epoch 8 Batch 150 Loss 2.4294 Accuracy 0.2610
Epoch 8 Batch 200 Loss 2.4191 Accuracy 0.2607
Epoch 8 Batch 250 Loss 2.4222 Accuracy 0.2614
Epoch 8 Batch 300 Loss 2.4258 Accuracy 0.2605
Epoch 8 Batch 350 Loss 2.4197 Accuracy 0.2611
Epoch 8 Batch 400 Loss 2.4185 Accuracy 0.2616
Epoch 8 Batch 450 Loss 2.4108 Accuracy 0.2626
Epoch 8 Batch 500 Loss 2.4083 Accuracy 0.2632
Epoch 8 Batch 550 Loss 2.4068 Accuracy 0.2632
Epoch 8 Batch 600 Loss 2.4052 Accuracy 0.2636
Epoch 8 Batch 650 Loss 2.4035 Accuracy 0.2640
Epoch 8 Batch 700 Loss 2.4026 Accuracy 0.2642
Epoch 8 Loss 2.4023 Accuracy 0.2642
Time taken for 1 epoch: 29.819046020507812 secs

Epoch 9 Batch 0 Loss 2.1758 Accuracy 0.2726
Epoch 9 Batch 50 Loss 2.1437 Accuracy 0.2787
Epoch 9 Batch 100 Loss 2.1325 Accuracy 0.2779
Epoch 9 Batch 150 Loss 2.1442 Accuracy 0.2798
Epoch 9 Batch 200 Loss 2.1494 Accuracy 0.2793
Epoch 9 Batch 250 Loss 2.1558 Accuracy 0.2781
Epoch 9 Batch 300 Loss 2.1563 Accuracy 0.2770
Epoch 9 Batch 350 Loss 2.1532 Accuracy 0.2776
Epoch 9 Batch 400 Loss 2.1518 Accuracy 0.2782
Epoch 9 Batch 450 Loss 2.1509 Accuracy 0.2788
Epoch 9 Batch 500 Loss 2.1491 Accuracy 0.2791
Epoch 9 Batch 550 Loss 2.1510 Accuracy 0.2791
Epoch 9 Batch 600 Loss 2.1549 Accuracy 0.2790
Epoch 9 Batch 650 Loss 2.1585 Accuracy 0.2788
Epoch 9 Batch 700 Loss 2.1615 Accuracy 0.2787
Epoch 9 Loss 2.1623 Accuracy 0.2787
Time taken for 1 epoch: 29.989798307418823 secs

Epoch 10 Batch 0 Loss 1.8333 Accuracy 0.3179
Epoch 10 Batch 50 Loss 1.9615 Accuracy 0.2914
Epoch 10 Batch 100 Loss 1.9492 Accuracy 0.2925
Epoch 10 Batch 150 Loss 1.9489 Accuracy 0.2918
Epoch 10 Batch 200 Loss 1.9501 Accuracy 0.2905
Epoch 10 Batch 250 Loss 1.9581 Accuracy 0.2901
Epoch 10 Batch 300 Loss 1.9624 Accuracy 0.2900
Epoch 10 Batch 350 Loss 1.9671 Accuracy 0.2892
Epoch 10 Batch 400 Loss 1.9662 Accuracy 0.2897
Epoch 10 Batch 450 Loss 1.9668 Accuracy 0.2898
Epoch 10 Batch 500 Loss 1.9694 Accuracy 0.2895
Epoch 10 Batch 550 Loss 1.9697 Accuracy 0.2894
Epoch 10 Batch 600 Loss 1.9728 Accuracy 0.2892
Epoch 10 Batch 650 Loss 1.9747 Accuracy 0.2889
Epoch 10 Batch 700 Loss 1.9797 Accuracy 0.2887
Saving checkpoint for epoch 10 at ./checkpoints/train/ckpt-2
Epoch 10 Loss 1.9796 Accuracy 0.2888
Time taken for 1 epoch: 30.037305116653442 secs

Epoch 11 Batch 0 Loss 1.6777 Accuracy 0.2772
Epoch 11 Batch 50 Loss 1.7663 Accuracy 0.3031
Epoch 11 Batch 100 Loss 1.7794 Accuracy 0.3029
Epoch 11 Batch 150 Loss 1.7946 Accuracy 0.3016
Epoch 11 Batch 200 Loss 1.7989 Accuracy 0.3003
Epoch 11 Batch 250 Loss 1.8028 Accuracy 0.2993
Epoch 11 Batch 300 Loss 1.8051 Accuracy 0.2994
Epoch 11 Batch 350 Loss 1.8139 Accuracy 0.2992
Epoch 11 Batch 400 Loss 1.8177 Accuracy 0.2997
Epoch 11 Batch 450 Loss 1.8216 Accuracy 0.2996
Epoch 11 Batch 500 Loss 1.8223 Accuracy 0.2997
Epoch 11 Batch 550 Loss 1.8277 Accuracy 0.2993
Epoch 11 Batch 600 Loss 1.8287 Accuracy 0.2991
Epoch 11 Batch 650 Loss 1.8330 Accuracy 0.2991
Epoch 11 Batch 700 Loss 1.8370 Accuracy 0.2988
Epoch 11 Loss 1.8371 Accuracy 0.2988
Time taken for 1 epoch: 29.774412155151367 secs

Epoch 12 Batch 0 Loss 1.4845 Accuracy 0.3057
Epoch 12 Batch 50 Loss 1.6447 Accuracy 0.3132
Epoch 12 Batch 100 Loss 1.6491 Accuracy 0.3093
Epoch 12 Batch 150 Loss 1.6589 Accuracy 0.3104
Epoch 12 Batch 200 Loss 1.6680 Accuracy 0.3096
Epoch 12 Batch 250 Loss 1.6757 Accuracy 0.3085
Epoch 12 Batch 300 Loss 1.6814 Accuracy 0.3074
Epoch 12 Batch 350 Loss 1.6881 Accuracy 0.3066
Epoch 12 Batch 400 Loss 1.6908 Accuracy 0.3066
Epoch 12 Batch 450 Loss 1.6932 Accuracy 0.3065
Epoch 12 Batch 500 Loss 1.6989 Accuracy 0.3061
Epoch 12 Batch 550 Loss 1.7042 Accuracy 0.3059
Epoch 12 Batch 600 Loss 1.7078 Accuracy 0.3060
Epoch 12 Batch 650 Loss 1.7145 Accuracy 0.3054
Epoch 12 Batch 700 Loss 1.7189 Accuracy 0.3052
Epoch 12 Loss 1.7191 Accuracy 0.3052
Time taken for 1 epoch: 29.73801016807556 secs

Epoch 13 Batch 0 Loss 1.6289 Accuracy 0.3061
Epoch 13 Batch 50 Loss 1.5185 Accuracy 0.3223
Epoch 13 Batch 100 Loss 1.5305 Accuracy 0.3205
Epoch 13 Batch 150 Loss 1.5481 Accuracy 0.3192
Epoch 13 Batch 200 Loss 1.5569 Accuracy 0.3182
Epoch 13 Batch 250 Loss 1.5692 Accuracy 0.3176
Epoch 13 Batch 300 Loss 1.5743 Accuracy 0.3168
Epoch 13 Batch 350 Loss 1.5804 Accuracy 0.3163
Epoch 13 Batch 400 Loss 1.5847 Accuracy 0.3158
Epoch 13 Batch 450 Loss 1.5887 Accuracy 0.3153
Epoch 13 Batch 500 Loss 1.5929 Accuracy 0.3151
Epoch 13 Batch 550 Loss 1.5984 Accuracy 0.3149
Epoch 13 Batch 600 Loss 1.6033 Accuracy 0.3143
Epoch 13 Batch 650 Loss 1.6082 Accuracy 0.3138
Epoch 13 Batch 700 Loss 1.6165 Accuracy 0.3133
Epoch 13 Loss 1.6167 Accuracy 0.3133
Time taken for 1 epoch: 29.787348985671997 secs

Epoch 14 Batch 0 Loss 1.4362 Accuracy 0.3707
Epoch 14 Batch 50 Loss 1.4458 Accuracy 0.3249
Epoch 14 Batch 100 Loss 1.4723 Accuracy 0.3225
Epoch 14 Batch 150 Loss 1.4770 Accuracy 0.3217
Epoch 14 Batch 200 Loss 1.4824 Accuracy 0.3209
Epoch 14 Batch 250 Loss 1.4857 Accuracy 0.3207
Epoch 14 Batch 300 Loss 1.4921 Accuracy 0.3202
Epoch 14 Batch 350 Loss 1.4959 Accuracy 0.3201
Epoch 14 Batch 400 Loss 1.4984 Accuracy 0.3205
Epoch 14 Batch 450 Loss 1.5051 Accuracy 0.3201
Epoch 14 Batch 500 Loss 1.5085 Accuracy 0.3200
Epoch 14 Batch 550 Loss 1.5126 Accuracy 0.3197
Epoch 14 Batch 600 Loss 1.5180 Accuracy 0.3191
Epoch 14 Batch 650 Loss 1.5230 Accuracy 0.3188
Epoch 14 Batch 700 Loss 1.5282 Accuracy 0.3189
Epoch 14 Loss 1.5284 Accuracy 0.3189
Time taken for 1 epoch: 29.765388250350952 secs

Epoch 15 Batch 0 Loss 1.3649 Accuracy 0.3081
Epoch 15 Batch 50 Loss 1.3745 Accuracy 0.3236
Epoch 15 Batch 100 Loss 1.3760 Accuracy 0.3253
Epoch 15 Batch 150 Loss 1.3856 Accuracy 0.3244
Epoch 15 Batch 200 Loss 1.3953 Accuracy 0.3237
Epoch 15 Batch 250 Loss 1.4052 Accuracy 0.3251
Epoch 15 Batch 300 Loss 1.4127 Accuracy 0.3244
Epoch 15 Batch 350 Loss 1.4212 Accuracy 0.3239
Epoch 15 Batch 400 Loss 1.4230 Accuracy 0.3249
Epoch 15 Batch 450 Loss 1.4271 Accuracy 0.3246
Epoch 15 Batch 500 Loss 1.4337 Accuracy 0.3240
Epoch 15 Batch 550 Loss 1.4392 Accuracy 0.3240
Epoch 15 Batch 600 Loss 1.4433 Accuracy 0.3239
Epoch 15 Batch 650 Loss 1.4486 Accuracy 0.3237
Epoch 15 Batch 700 Loss 1.4536 Accuracy 0.3235
Saving checkpoint for epoch 15 at ./checkpoints/train/ckpt-3
Epoch 15 Loss 1.4537 Accuracy 0.3235
Time taken for 1 epoch: 30.007831811904907 secs

Epoch 16 Batch 0 Loss 1.4324 Accuracy 0.3477
Epoch 16 Batch 50 Loss 1.3043 Accuracy 0.3380
Epoch 16 Batch 100 Loss 1.3125 Accuracy 0.3351
Epoch 16 Batch 150 Loss 1.3182 Accuracy 0.3330
Epoch 16 Batch 200 Loss 1.3266 Accuracy 0.3320
Epoch 16 Batch 250 Loss 1.3326 Accuracy 0.3323
Epoch 16 Batch 300 Loss 1.3372 Accuracy 0.3321
Epoch 16 Batch 350 Loss 1.3475 Accuracy 0.3310
Epoch 16 Batch 400 Loss 1.3536 Accuracy 0.3309
Epoch 16 Batch 450 Loss 1.3599 Accuracy 0.3300
Epoch 16 Batch 500 Loss 1.3651 Accuracy 0.3302
Epoch 16 Batch 550 Loss 1.3705 Accuracy 0.3297
Epoch 16 Batch 600 Loss 1.3744 Accuracy 0.3294
Epoch 16 Batch 650 Loss 1.3806 Accuracy 0.3289
Epoch 16 Batch 700 Loss 1.3863 Accuracy 0.3285
Epoch 16 Loss 1.3865 Accuracy 0.3285
Time taken for 1 epoch: 29.82802104949951 secs

Epoch 17 Batch 0 Loss 1.1936 Accuracy 0.3390
Epoch 17 Batch 50 Loss 1.2649 Accuracy 0.3309
Epoch 17 Batch 100 Loss 1.2614 Accuracy 0.3355
Epoch 17 Batch 150 Loss 1.2714 Accuracy 0.3358
Epoch 17 Batch 200 Loss 1.2779 Accuracy 0.3350
Epoch 17 Batch 250 Loss 1.2837 Accuracy 0.3343
Epoch 17 Batch 300 Loss 1.2900 Accuracy 0.3337
Epoch 17 Batch 350 Loss 1.2942 Accuracy 0.3331
Epoch 17 Batch 400 Loss 1.2970 Accuracy 0.3336
Epoch 17 Batch 450 Loss 1.3005 Accuracy 0.3336
Epoch 17 Batch 500 Loss 1.3061 Accuracy 0.3335
Epoch 17 Batch 550 Loss 1.3107 Accuracy 0.3339
Epoch 17 Batch 600 Loss 1.3165 Accuracy 0.3334
Epoch 17 Batch 650 Loss 1.3211 Accuracy 0.3330
Epoch 17 Batch 700 Loss 1.3270 Accuracy 0.3328
Epoch 17 Loss 1.3270 Accuracy 0.3328
Time taken for 1 epoch: 30.348626852035522 secs

Epoch 18 Batch 0 Loss 1.2057 Accuracy 0.3574
Epoch 18 Batch 50 Loss 1.1891 Accuracy 0.3427
Epoch 18 Batch 100 Loss 1.1974 Accuracy 0.3434
Epoch 18 Batch 150 Loss 1.2029 Accuracy 0.3414
Epoch 18 Batch 200 Loss 1.2115 Accuracy 0.3407
Epoch 18 Batch 250 Loss 1.2209 Accuracy 0.3405
Epoch 18 Batch 300 Loss 1.2263 Accuracy 0.3394
Epoch 18 Batch 350 Loss 1.2310 Accuracy 0.3393
Epoch 18 Batch 400 Loss 1.2381 Accuracy 0.3390
Epoch 18 Batch 450 Loss 1.2429 Accuracy 0.3384
Epoch 18 Batch 500 Loss 1.2515 Accuracy 0.3381
Epoch 18 Batch 550 Loss 1.2567 Accuracy 0.3375
Epoch 18 Batch 600 Loss 1.2625 Accuracy 0.3374
Epoch 18 Batch 650 Loss 1.2679 Accuracy 0.3369
Epoch 18 Batch 700 Loss 1.2724 Accuracy 0.3367
Epoch 18 Loss 1.2726 Accuracy 0.3367
Time taken for 1 epoch: 29.817647457122803 secs

Epoch 19 Batch 0 Loss 1.0279 Accuracy 0.3442
Epoch 19 Batch 50 Loss 1.1284 Accuracy 0.3486
Epoch 19 Batch 100 Loss 1.1431 Accuracy 0.3494
Epoch 19 Batch 150 Loss 1.1595 Accuracy 0.3468
Epoch 19 Batch 200 Loss 1.1620 Accuracy 0.3449
Epoch 19 Batch 250 Loss 1.1672 Accuracy 0.3459
Epoch 19 Batch 300 Loss 1.1739 Accuracy 0.3448
Epoch 19 Batch 350 Loss 1.1834 Accuracy 0.3436
Epoch 19 Batch 400 Loss 1.1858 Accuracy 0.3439
Epoch 19 Batch 450 Loss 1.1926 Accuracy 0.3432
Epoch 19 Batch 500 Loss 1.1973 Accuracy 0.3430
Epoch 19 Batch 550 Loss 1.2031 Accuracy 0.3426
Epoch 19 Batch 600 Loss 1.2106 Accuracy 0.3425
Epoch 19 Batch 650 Loss 1.2173 Accuracy 0.3421
Epoch 19 Batch 700 Loss 1.2223 Accuracy 0.3414
Epoch 19 Loss 1.2226 Accuracy 0.3413
Time taken for 1 epoch: 29.91853427886963 secs

Epoch 20 Batch 0 Loss 1.0190 Accuracy 0.3329
Epoch 20 Batch 50 Loss 1.1070 Accuracy 0.3518
Epoch 20 Batch 100 Loss 1.1074 Accuracy 0.3502
Epoch 20 Batch 150 Loss 1.1073 Accuracy 0.3508
Epoch 20 Batch 200 Loss 1.1195 Accuracy 0.3490
Epoch 20 Batch 250 Loss 1.1232 Accuracy 0.3487
Epoch 20 Batch 300 Loss 1.1320 Accuracy 0.3483
Epoch 20 Batch 350 Loss 1.1363 Accuracy 0.3478
Epoch 20 Batch 400 Loss 1.1414 Accuracy 0.3478
Epoch 20 Batch 450 Loss 1.1493 Accuracy 0.3472
Epoch 20 Batch 500 Loss 1.1573 Accuracy 0.3458
Epoch 20 Batch 550 Loss 1.1636 Accuracy 0.3454
Epoch 20 Batch 600 Loss 1.1691 Accuracy 0.3449
Epoch 20 Batch 650 Loss 1.1744 Accuracy 0.3443
Epoch 20 Batch 700 Loss 1.1783 Accuracy 0.3441
Saving checkpoint for epoch 20 at ./checkpoints/train/ckpt-4
Epoch 20 Loss 1.1786 Accuracy 0.3441
Time taken for 1 epoch: 30.05885624885559 secs
"""

#Evaluate

def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                            if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_en.decode([i for i in result
                                              if i < tokenizer_en.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)




