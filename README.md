# Portugese-to-English-Translation
A transformer model is trained to translate Portugese to English

The core idea behind the Transformer model is self-attention—the ability to attend to different positions of the input sequence to compute a representation of that sequence. Transformer creates stacks of self-attention layers.

A transformer model handles variable-sized input using stacks of self-attention layers instead of RNNs or CNNs. This general architecture has a number of advantages:

1. It make no assumptions about the temporal/spatial relationships across the data. This is ideal for processing a set of objects (for example, StarCraft units).
2. Layer outputs can be calculated in parallel, instead of a series like an RNN.
3. Distant items can affect each other's output without passing through many RNN-steps, or convolution layers (see Scene Memory Transformer for example).
4. It can learn long-range dependencies. This is a challenge in many sequence tasks.

The downsides of this architecture are:

1. For a time-series, the output for a time-step is calculated from the entire history instead of only the inputs and current hidden-state. This may be less efficient.
2. If the input does have a temporal/spatial relationship, like text, some positional encoding must be added or the model will effectively see a bag of words.

# The dataset
We use TFDS (https://www.tensorflow.org/datasets) to load the Portugese-English translation dataset (https://github.com/neulab/word-embeddings-for-nmt) from the TED Talks Open Translation Project (https://www.ted.com/participate/translate).

This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.

# Data Preprocessing
The training dataset is tokenized using sub-words. 
An Example:  
Tokenized string is [7915, 1248, 7946, 7194, 13, 2799, 7877]  
The original string: Transformer is awesome.  
The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary.  
7915 ----> T  
1248 ----> ran  
7946 ----> s  
7194 ----> former   
13 ----> is   
2799 ----> awesome  
7877 ----> .  

# Positional encoding
Positional encoding is added to give the model some information about the relative position of the words in the sentence. The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be closer to each other. But the embeddings do not encode the relative position of words in a sentence. So after adding the positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sentence, in the d-dimensional space.  

# Masking
Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise. The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.

# Scaled dot product attention
 
![alt text](https://github.com/MedentzidisCharalampos/Portugese-to-English-Translation-using-Transformer-Model/blob/main/scaled_attention.png)  
 
The attention function used by the transformer takes three inputs: Q (query), K (key), V (value).  The equation used to calculate the attention weights is:

![alt text](https://github.com/MedentzidisCharalampos/Portugese-to-English-Translation-using-Transformer-Model/blob/main/attention_equation.png)  

The dot-product attention is scaled by a factor of square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients resulting in a very hard softmax.

For example, consider that Q and K have a mean of 0 and variance of 1. Their matrix multiplication will have a mean of 0 and variance of dk. Hence, square root of dk is used for scaling (and not any other number) because the matmul of Q and K should have a mean of 0 and variance of 1, and you get a gentler softmax.

The mask is multiplied with -1e9 (close to negative infinity). This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately before a softmax. The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.

# Multi-head attention

![alt text](https://github.com/MedentzidisCharalampos/Portugese-to-English-Translation-using-Transformer-Model/blob/main/multi_head_attention.png)  


Multi-head attention consists of four parts:

1. Linear layers and split into heads.
2. Scaled dot-product attention.
3. Concatenation of heads.
4. Final linear layer.
5. Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear (Dense) layers and split up into multiple heads.

Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to jointly attend to information at different positions from different representational spaces. After the split each head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full dimensionality.

Point wise feed forward network

Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.


# Encoder and decoder

![alt text](https://github.com/MedentzidisCharalampos/Portugese-to-English-Translation-using-Transformer-Model/blob/main/scaled_attention.png)  

1. The input sentence is passed through N encoder layers that generates an output for each word/token in the sequence.
2. The decoder attends on the encoder's output and its own input (self-attention) to predict the next word.

Encoder layer
Each encoder layer consists of sublayers:

Multi-head attention (with padding mask)
Point wise feed forward networks.
Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections help in avoiding the vanishing gradient problem in deep networks.

The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis. There are N encoder layers in the transformer.

Decoder layer
Each decoder layer consists of sublayers:

Masked multi-head attention (with look ahead mask and padding mask)
Multi-head attention (with padding mask). V (value) and K (key) receive the encoder output as inputs. Q (query) receives the output from the masked multi-head attention sublayer.
Point wise feed forward networks
Each of these sublayers has a residual connection around it followed by a layer normalization. The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis.

There are N decoder layers in the transformer.

As Q receives the output from decoder's first attention block, and K receives the encoder output, the attention weights represent the importance given to the decoder's input based on the encoder's output. In other words, the decoder predicts the next word by looking at the encoder output and self-attending to its own output. See the demonstration above in the scaled dot product attention section.


Encoder
The Encoder consists of:

Input Embedding
Positional Encoding
N encoder layers
The input is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the encoder layers. The output of the encoder is the input to the decoder.

Decoder
The Decoder consists of:

Output Embedding
Positional Encoding
N decoder layers
The target is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the decoder layers. The output of the decoder is the input to the final linear layer.

Create the Transformer
Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the input to the linear layer and its output is returned.

Evaluate
The following steps are used for evaluation:

Encode the input sentence using the Portuguese tokenizer (tokenizer_pt). Moreover, add the start and end token so the input is equivalent to what the model is trained with. This is the encoder input.
The decoder input is the start token == tokenizer_en.vocab_size.
Calculate the padding masks and the look ahead masks.
The decoder then outputs the predictions by looking at the encoder output and its own output (self-attention).
Select the last word and calculate the argmax of that.
Concatentate the predicted word to the decoder input as pass it to the decoder.
In this approach, the decoder predicts the next word based on the previous words it predicted.
