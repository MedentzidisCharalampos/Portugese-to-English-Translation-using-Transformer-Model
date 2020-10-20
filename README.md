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
