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

# Input Pipeline
We use TFDS (https://www.tensorflow.org/datasets) to load the Portugese-English translation dataset (https://github.com/neulab/word-embeddings-for-nmt) from the TED Talks Open Translation Project (https://www.ted.com/participate/translate).

This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.

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

Positional encoding
Positional encoding is added to give the model some information about the relative position of the words in the sentence. The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be closer to each other. But the embeddings do not encode the relative position of words in a sentence. So after adding the positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sentence, in the d-dimensional space.  

Masking
Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.

Scaled dot product attention
 
 ![alt text](https://github.com/MedentzidisCharalampos/Portugese-to-English-Translation-using-Transformer-Model/blob/main/scaled_attention.png)  
 
 The attention function used by the transformer takes three inputs: Q (query), K (key), V (value).
 
 The dot-product attention is scaled by a factor of square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients resulting in a very hard softmax.

For example, consider that Q and K have a mean of 0 and variance of 1. Their matrix multiplication will have a mean of 0 and variance of dk. Hence, square root of dk is used for scaling (and not any other number) because the matmul of Q and K should have a mean of 0 and variance of 1, and you get a gentler softmax.

The mask is multiplied with -1e9 (close to negative infinity). This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately before a softmax. The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.
