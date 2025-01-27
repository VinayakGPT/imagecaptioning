import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN = False): # Setup
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images): # Action
        features = self.inception(images)
        # Inception model returns a named tuple (logits, aux_logits), we only need logits
        if isinstance(features, models.inception.InceptionOutputs):
            features = features.logits

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True # We are essentially training the last layer, and not the rest.
            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.feature_layer = nn.Linear(feature_dim, attention_dim)  # For image features
        self.hidden_layer = nn.Linear(hidden_dim, attention_dim)   # For decoder hidden state
        self.attention_layer = nn.Linear(attention_dim, 1)         # For combined attention
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden):
        # Compute attention scores
        features_attention = self.feature_layer(features)           # (batch_size, num_pixels, attention_dim)
        hidden_attention = self.hidden_layer(hidden).unsqueeze(1)   # (batch_size, 1, attention_dim)
        combined_attention = self.relu(features_attention + hidden_attention)
        attention_scores = self.attention_layer(combined_attention).squeeze(2)  # (batch_size, num_pixels)
        attention_weights = self.softmax(attention_scores)          # (batch_size, num_pixels)

        # Weighted sum of features
        weighted_features = (features * attention_weights.unsqueeze(2)).sum(dim=1)  # (batch_size, feature_dim)
        return weighted_features, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # It's going to take an index and map it to some embed_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length = 50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption] # Return string and not indices as final output
