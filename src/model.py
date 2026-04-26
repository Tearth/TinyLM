import torch
import bidict

from torch import nn
from torch import Tensor
from bidict import bidict

class Model(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, context_size: int, transformers_count: int, ff_network_size: int) -> None:
        super().__init__()
        
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.transformers_count = transformers_count

        self.tokenizer = Tokenizer()
        self.encoder_decoder = EncoderDecoder()
        self.embedding_layer = EmbeddingLayer(vocabulary_size, embedding_size)
        self.position_encoding_layer = PositionEncodingLayer(context_size, embedding_size)
        self.transformers = nn.ModuleList([
            TransformerLayer(ff_network_size, embedding_size) for _ in range(transformers_count)
        ])
        self.output_layer = OutputLayer(vocabulary_size, embedding_size)

    def prompt(self, input: str) -> str:
        print("Input:", input)

        # Split input into a list of separate tokens, where each one represents a single lower-cased word - no interpunction supported for now
        tokens = self.tokenizer.get_tokens(input)
        token_ids = self.encoder_decoder.encode_list(tokens)

        print("Tokens:", ", ".join(tokens))
        print("Token IDs:", ", ".join(map(str, token_ids)))

        input_tensor = torch.tensor(token_ids).unsqueeze(0)
        output_tensor = self(input_tensor)

        print("Model Input:", input_tensor)
        print("Model Output:")
        print(output_tensor)

        output_vector = output_tensor[:, -1, :]
        probability_vector = torch.softmax(output_vector, dim=-1)
        next_token_index = int(torch.argmax(probability_vector).item())
        next_token = self.encoder_decoder.decode(next_token_index)

        print()
        print("Output Vector:", output_vector)
        print("Probability Vector:", probability_vector)
        print("Best Candidates:")

        sorted = torch.sort(probability_vector, descending=True)

        for i in range(3):
            index = int(sorted.indices[0][i].item())
            token = self.encoder_decoder.decode(index)
            probability = float(probability_vector[0][index].item())
            print(f"  {token} ({probability * 100.0:0.2f}%)")

        return next_token
    
    def forward(self, x : Tensor) -> Tensor:
        x = self.embedding_layer(x)
        x = self.position_encoding_layer(x)

        for transformer in self.transformers:
            x = transformer(x)
        x = self.output_layer(x)

        return x
    
class Tokenizer:
    def __init__(self) -> None:
        pass

    def get_tokens(self, input: str) -> list[str]:
        return input.lower().split()
    
class EncoderDecoder:
    def __init__(self) -> None:
        self.map = bidict({
            'floppa1': 0,
            'floppa2': 1,
            'floppa3': 2
        })

    def encode(self, token: str) -> int:
        return self.map[token]
    
    def encode_list(self, tokens: list[str]) -> list[int]:
        result = []

        for token in tokens:
            result.append(self.encode(token))
        
        return result
    
    def decode(self, id: int) -> str:
        return self.map.inverse[id]
    
class EmbeddingLayer(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int) -> None:
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Each token has unique vector of `embedding_size` length
        self.embedding_matrix = nn.Embedding(vocabulary_size, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        # Input: [batch; sequence_size]
        # Output: [batch; sequence_size; embedding_size]
        
        # [ 3, 1, 2]
        #
        # becomes
        #
        # [0.1; 0.2; 0.3]
        # [0.2; 0.3; 0.4]
        # [0.3; 0.4; 0.5]
        return self.embedding_matrix(x)
    
class PositionEncodingLayer(nn.Module):
    def __init__(self, context_size: int, embedding_size: int) -> None:
        super().__init__()
        
        self.context_size = context_size
        self.embedding_size = embedding_size

        # Each context position has unique vector of `embedding_size` length
        self.position_matrix = nn.Embedding(context_size, embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        sequence = torch.arange(x.size(1), device=x.device)
        position_encoding = self.position_matrix(sequence)

        # [0.1; 0.2; 0.3]   [0.1; 0.1; 0.1]   [0.2; 0.3; 0.4]
        # [0.2; 0.3; 0.4] + [0.2; 0.2; 0.2] = [0.4; 0.5; 0.6]
        # [0.3; 0.4; 0.5]   [0.3; 0.3; 0.3]   [0.6; 0.7; 0.8]
        return x + position_encoding.unsqueeze(0)

class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        
        self.embedding_size = embedding_size

        # Three matrices to represent Query, Key and Value
        # Query - what tokens are looking for
        # Key - what tokens are representing
        # Value - what tokens are providing
        self.q_matrix = nn.Linear(embedding_size, embedding_size)
        self.k_matrix = nn.Linear(embedding_size, embedding_size)
        self.v_matrix = nn.Linear(embedding_size, embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        # Embedded and position encoded tokens are projected into Q, K and V matrices
        q = self.q_matrix(x) # [batch; sequence_size; embedding_size]
        k = self.k_matrix(x) # [batch; sequence_size; embedding_size]
        v = self.v_matrix(x) # [batch; sequence_size; embedding_size]

        # Mask to cover information about tokens after the currently processed one
        # [ 0, 1, 1 ]
        # [ 0, 0, 1 ]
        # [ 0, 0, 0 ]
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1)
        mask = mask.unsqueeze(0)

        # Calculate attention weights by multiplying Query and Key
        attention_weights = q @ torch.transpose(k, -2, -1)

        # Mask is multiplied by -1e9 and added to attention weights, so masked ones will be near zero after softmax
        attention_weights += mask * -1e9

        # Attention weights are scaled, so the sum of probability is equal to 1.0
        attention_scaled = torch.softmax(attention_weights / (self.embedding_size ** 0.5), dim=-1)

        # Attention with scaled weights is multiplied with Value to get the final ones
        attention_value = attention_scaled @ v

        return attention_value
    
class FeedForwardNetworkLayer(nn.Module):
    def __init__(self, ff_network_size: int, embedding_size: int) -> None:
        super().__init__()

        self.ff_network_size = ff_network_size
        self.embedding_size = embedding_size

        self.layer_a = nn.Linear(embedding_size, ff_network_size)
        self.layer_b = nn.Linear(ff_network_size, embedding_size)
        self.activation = nn.ReLU()
    
    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        x = self.layer_a(x)
        x = self.activation(x)
        x = self.layer_b(x)

        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, ff_network_size: int, embedding_size: int) -> None:
        super().__init__()

        self.attention_layer = SelfAttentionLayer(embedding_size)
        self.attention_norm = nn.LayerNorm(embedding_size)

        self.ff_network_layer = FeedForwardNetworkLayer(ff_network_size, embedding_size)
        self.ff_network_norm = nn.LayerNorm(embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        x = x + self.attention_norm(self.attention_layer(x))
        x = x + self.ff_network_layer(self.ff_network_norm(x))

        return x
    
class OutputLayer(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.output_matrix = nn.Linear(embedding_size, vocabulary_size)
        self.output_norm = nn.LayerNorm(embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; vocabulary_size]

        return self.output_matrix(self.output_norm(x))
