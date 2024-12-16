import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from opacus.layers.dp_rnn import RNNLinear, DPLSTM
import torch.nn as nn

# credit https://www.kaggle.com/code/m0hammadjavad/imdb-sentiment-classifier-pytorch
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers):
        super(SentimentModel, self).__init__()

        # embedding layer maps each word (represented as an integer) to a dense vector of a fixed size (embed_size). 
        # It essentially converts word indices into word vectors that capture semantic meanings.
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # lstm layer is used for processing sequential data. 
        # It's particularly effective for tasks like text processing because it can capture long-range dependencies in the data.
        self.lstm = DPLSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # The input x is a batch of sequences (movie reviews) where each word is represented by an index (from the vocabulary). 
        # The embedding layer converts these indices into word vectors (embeddings), 
        # resulting in a tensor of shape (batch_size, sequence_length, embed_size)
        embeds = self.embedding(x)

        # The embeddings are passed into the LSTM layer, which processes the sequence of word vectors. 
        # The LSTM produces an output for each time step (word) in the sequence. 
        # The output is of shape (batch_size, sequence_length, hidden_size), w
        # here each word in the sequence has an associated hidden state vector.
        lstm_out, _ = self.lstm(embeds)

        # Here, we are only interested in the last hidden state of the LSTM. 
        # This is because, for sentiment classification, 
        # we typically assume that the final state of the LSTM summarizes the entire sequence (review). 
        # This line extracts the last hidden state from the output, which has a shape of (batch_size, hidden_size).
        lstm_out = lstm_out[:, -1, :]

        # The last hidden state is passed through the fully connected (linear) layer, which maps it to a single value (since output_size = 1). 
        # This value represents the raw prediction for the sentiment (before applying an activation function).
        out = self.fc(lstm_out)

        # Finally, we apply the sigmoid function to the output. 
        # The sigmoid function maps the raw output to a value between 0 and 1, 
        # which can be interpreted as the probability of the review being positive. 
        # If the probability is greater than 0.5, the model predicts "positive"; otherwise, it predicts "negative".
        return torch.sigmoid(out)
    
class SentimentModelNoEmbed(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers):
        super(SentimentModelNoEmbed, self).__init__()
        # lstm layer is used for processing sequential data. 
        # It's particularly effective for tasks like text processing because it can capture long-range dependencies in the data.
        self.lstm = DPLSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # The input x is a batch of sequences (movie reviews) where each word is represented by an index (from the vocabulary). 
        # The embedding layer converts these indices into word vectors (embeddings), 
        # resulting in a tensor of shape (batch_size, sequence_length, embed_size)

        # The embeddings are passed into the LSTM layer, which processes the sequence of word vectors. 
        # The LSTM produces an output for each time step (word) in the sequence. 
        # The output is of shape (batch_size, sequence_length, hidden_size), w
        # here each word in the sequence has an associated hidden state vector.
        lstm_out, _ = self.lstm(x)

        # Here, we are only interested in the last hidden state of the LSTM. 
        # This is because, for sentiment classification, 
        # we typically assume that the final state of the LSTM summarizes the entire sequence (review). 
        # This line extracts the last hidden state from the output, which has a shape of (batch_size, hidden_size).
        lstm_out = lstm_out[:, -1, :]

        # The last hidden state is passed through the fully connected (linear) layer, which maps it to a single value (since output_size = 1). 
        # This value represents the raw prediction for the sentiment (before applying an activation function).
        out = self.fc(lstm_out)

        # Finally, we apply the sigmoid function to the output. 
        # The sigmoid function maps the raw output to a value between 0 and 1, 
        # which can be interpreted as the probability of the review being positive. 
        # If the probability is greater than 0.5, the model predicts "positive"; otherwise, it predicts "negative".
        return torch.sigmoid(out)



class fcNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Use DPLinear layers for better compatibility with Opacus
        self.layers = nn.Sequential(
            nn.Flatten(),  # Flatten input first
            RNNLinear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),  # inplace=True can help with memory
            RNNLinear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            RNNLinear(hidden_dim, output_dim)
        )
        
        # Optional: Add weight normalization or initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, RNNLinear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Normalize input (if needed)
        x = (x - 0.1307) / 0.3081
        
        # Pass through the sequential layers
        return self.layers(x)
    


class MembershipInferenceAttacker(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MembershipInferenceAttacker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = (x - 0.1307)/0.3081
        x = nn.Flatten()(x)
        x = self.network(x)
        return x