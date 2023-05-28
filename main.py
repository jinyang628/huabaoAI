import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow_hub as hub
import jieba
import numpy as np
import tarfile
import gensim

# Read the dataset from the text file
with open('huabao-data.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create empty lists for source and target sentences
source_sentences = []
target_sentences = []

# Process each line in the dataset
for line in lines:
    sentences = line.strip().split('\t')
    # jieba.cut applies various segmentation algorithms to identify the boundaries between words and
    # returns an iterable object that yields the segmented words
    source_sentences.append(' '.join(jieba.cut(sentences[1])))
    target_sentences.append(' '.join(jieba.cut(sentences[2])))

# Create a DataFrame from the lists
df = pd.DataFrame({'source': source_sentences, 'target': target_sentences})

# Remove any rows with missing values
df.dropna(inplace=True)

# Split the dataset into training and validation sets
# 20% used for testing with 42 being the seed value for randomisation
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Print the sizes of the training and validation sets
# print("Training set size:", len(train_df))
# print("Validation set size:", len(val_df))

# Tokenization
# Tokenizer is used to vectorize text data by converting it into sequences of integers
tokenizer = Tokenizer()
# Fits the tokenizer on the training data and updates the internal vocabulary of the tokenizer
# based on the text data provided. Assigns a unique integer index to each word based on its frequency
tokenizer.fit_on_texts(train_df['source'])

"""
Convert sentences into sequences
Takes the sentences from the "source" column of the training DataFrame and converts them into sequences of integers.
Each unique word in the sentences is assigned a unique integer value based on the tokenizer's vocabulary.
The resulting sequences represent the sentences in a numerical format that can be used as input to a neural network.
"""
source_sequences = tokenizer.texts_to_sequences(train_df['source'])

#  The purpose of having separate source and target sequences is typically for training a sequence-to-sequence model,
#  where the source represents the input sequence and the target represents the desired output or translation.
target_sequences = tokenizer.texts_to_sequences(train_df['target'])

# By using the same tokenizer, we prepare the "source" test data in the same format as the training data,
# making it suitable for evaluation or prediction with the trained model.
test_sequences = tokenizer.texts_to_sequences(test_df['source'])

# Padding
# Pad 0s to all the sequences at the end so that they are of the same length for training and prediction
max_length = max(max(len(seq) for seq in source_sequences),
                 max(len(seq) for seq in target_sequences))
padded_source_sequences = pad_sequences(source_sequences, maxlen=max_length, padding='post')
padded_target_sequences = pad_sequences(target_sequences, maxlen=max_length, padding='post')
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

"""
# I USED THE WRONG MODEL :(
# model_archive_path = './tfhub_module_dir/nnlm-en-dim50_2.tar.gz'
extract_dir = './tfhub_module_dir/extracted_model'
# Extract the contents of the archive to a directory
# with tarfile.open(model_archive_path, 'r:gz') as tar:
    # tar.extractall(extract_dir)

# Load the model from the extracted directory
embeddings = hub.load(extract_dir)
"""

embedding_dim = 100  # Dimensionality of the word embeddings
hidden_units = 256  # Number of units in the LSTM layers

"""
Calculates the vocabulary size for your model. tokenizer.word_index is a dictionary that maps words to their 
corresponding unique integer indices. len(tokenizer.word_index) returns the total number of unique words in your 
training data. Adding 1 accounts for an extra index reserved for out-of-vocabulary (OOV) words.
"""
vocab_size = len(tokenizer.word_index) + 1

"""
An encoder is a component that takes an input sequence and processes it into a fixed-dimensional representation or 
context vector. The encoder's purpose is to capture the input sequence's relevant information and compress it into a 
meaningful representation that can be used by the decoder

The encoder operates on the input sequence one element at a time, updating its internal hidden state at each step. 
The final hidden state of the encoder, which contains a summary of the input sequence, is passed to the decoder.
"""

"""
Defines the input layer for the encoder in your model. Specifies that the input will be a sequence of integers with a
length of max_length, representing the encoded source sentences.
"""
encoder_inputs = Input(shape=(max_length,))

"""
Embedding layer to the encoder. The embedding layer is responsible for mapping the input sequence of integers to their 
corresponding dense vectors. input_dim is set to vocab_size, indicating the size of the vocabulary. 
output_dim is set to embedding_dim, which defines the dimensionality of the embedding vectors.

With the embedding layer, the semantic meaning of each token is captured.
"""
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)

"""
These lines create the first LSTM layer of the encoder. hidden_units specifies the number of hidden units in the LSTM 
layer. return_sequences=True is set to return the full sequence of outputs, not just the last output. 
return_state=True is set to return the final hidden state and cell state of the LSTM.

The output of encoder_lstm1 is encoder_outputs1, which represents the output sequences from the LSTM layer. 
state_h1 and state_c1 represent the final hidden state and cell state of the LSTM layer.

Hidden state represents the RNN's memory or the summarised information from previous time steps. At each time step, the 
RNN's hidden state is updated based on the current input and previous hidden state. The hidden state is used to encode 
the context from the previous time steps, which helps the RNN capture long-term dependencies in the sequence. The hidden
state of the encoder is passed to the decoder to provide a starting point for generating the output sequence

Cell state is specific to LSTM units and serves as a support for the hidden state. At each time step, the cell state 
is updated using the current input, the previous cell state and the current hidden state. Acting as a memory unit, it 
selectively retains or forgets information based on the input and its relevance to the current time step. The cell state 
allows LSTM units to handle the vanishing gradient problem better than traditional RNNs, enabling them to capture 
long-term dependencies effectively 
"""
encoder_lstm1 = LSTM(hidden_units, return_sequences=True, return_state=True)
encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)

"""
The input to this layer is encoder_outputs1, which is the output sequence from the first LSTM layer
Chaining multiple LSTM layers in a sequence-to-sequence model serves the purpose of capturing more complex 
patterns in the input sequence. By stacking multiple LSTM layers, the higher layers receive the output sequences from 
the lower layers as their input. This allows the higher layers to operate on increasingly abstract representations 
of the input sequence, potentially capturing higher-level patterns and dependencies.

Main benefits of stacking multiple LSTM layers:
    1. Increased Model Capacity: Each LSTM layer in the stack introduces additional learnable parameters, 
    enabling the model to learn more complex representations of the input sequence. This increased capacity can 
    be beneficial when the input sequence exhibits complex dependencies or when the task requires capturing intricate 
    patterns.

    2. Hierarchical Representation Learning: The lower LSTM layers capture lower-level temporal dependencies, 
    such as short-term patterns, while the higher layers can capture longer-term dependencies and more abstract 
    representations. This hierarchical representation learning allows the model to capture both local and global 
    patterns in the input sequence.

However, stacking LSTM layers is a design choice and depends on the complexity of the task, the nature of the input 
sequence, and the available computational resources. Adding more layers increases the model's capacity but also requires 
more computational resources and may increase the risk of overfitting if not carefully regularized.
"""
encoder_lstm2 = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm2(encoder_outputs1)

# Contains the final hidden state and cell state of the encoder, which will be passed as initial states to the decoder.
encoder_states = [state_h, state_c]

"""
While the encoder processes the input sequence and extracts its meaningful representation, the decoder uses this 
representation to generate the desired output sequence.

Define the input for the decoder. The shape of the input is (max_length - 1,), which means that the decoder will 
receive sequences of length max_length - 1. The -1 is used because in sequence-to-sequence models, the decoder takes 
the target sequence as input, but shifts it by one position. This is done to predict the next word in the target 
sequence based on the previous words.
"""
decoder_inputs = Input(shape=(max_length - 1,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm1 = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_lstm2 = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm2(decoder_lstm1(decoder_embedding, initial_state=encoder_states))

"""
A dense layer with vocab_size units is added to the model, followed by a softmax activation function. 
This dense layer maps the output sequences from the LSTM layers to the vocabulary size, representing the probabilities 
of each word in the vocabulary that will be used to generate the predicted output. 

A dense layer, also known as a fully connected layer, is a type of neural network layer where each neuron is connected 
to every neuron in the previous layer. In a dense layer, every input is connected to every output by a weight, 
and these weights are learned during the training process. The primary purpose of a dense layer is to introduce 
non-linearity and enable the neural network to learn complex relationships between the input and output. It transforms 
the input data from the previous layer into a higher-dimensional space, allowing the network to capture and represent 
more abstract features and patterns.

The dense layer performs two main computations:
    1. Linear Transformation: Each neuron in the dense layer receives inputs from all the neurons in the previous layer.
    It computes a weighted sum of these inputs by multiplying the input values with their corresponding weights and 
    summing them up. The linear transformation can be represented by the equation: y = Wx + b, where y is the output, 
    W is the weight matrix, x is the input vector, and b is the bias vector.

    2. Activation Function: After the linear transformation, an activation function is applied element-wise to the 
    output of each neuron. The activation function introduces non-linearity into the network, allowing it to model 
    complex relationships.

The number of neurons in a dense layer determines the dimensionality of the output space. More neurons allow for a 
higher capacity to represent complex relationships, but also increase the number of trainable parameters in the network.

Dense layers are typically used in the final layers of a neural network to map the high-dimensional feature 
representations learned by previous layers to the desired output format.
"""
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
batch_size = 32

history = model.fit(
    [padded_source_sequences, padded_target_sequences[:, :-1]],
    padded_target_sequences[:, 1:],
    batch_size=batch_size,
    epochs=7,
    validation_split=0.2
)

# loss refers to loss function -> measures the discrepancy between the predicted output and the actual output
# accuracy -> represents the proportion of correctly predicted outputs compared to the total number of training examples
# val_loss refers to validation loss -> indicates how well the model is performing on unseen data
# val_accuracy -> represents the proportion of correctly predicted outputs on the validation set

# Testing
test_loss, test_accuracy = model.evaluate(
    [padded_test_sequences, padded_test_sequences[:, :-1]],
    padded_test_sequences[:, 1:]
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Prediction
input_sentences = ['我苹果喜欢', '我中国在住']
input_sequences = tokenizer.texts_to_sequences(input_sentences)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')

predictions = model.predict([padded_input_sequences, padded_input_sequences[:, :-1]])

corrected_sentences = []
for prediction in predictions:
    corrected_sentence = ' '.join([tokenizer.index_word.get(index, '') for index in np.argmax(prediction, axis=1)])
    corrected_sentences.append(corrected_sentence)

for input_sentence, corrected_sentence in zip(input_sentences, corrected_sentences):
    print('Input:', input_sentence)
    print('Corrected:', corrected_sentence)
    print()

#comma and 的 keep appearing in the prediction -> might be because they are the most common words in the dataset
