import torch

from torch import device, nn
from torch import Tensor
from tokens import TokenDictionary
from typing import Any

class ModelPersistence:
    def __init__(
        self, state: dict[str, Any],
        token_dictionary: TokenDictionary,
        vocabulary_size: int,
        embedding_size: int,
        context_size: int,
        transformers_count: int,
        ff_network_size: int
    ) -> None:
        self.state = state
        self.token_dictionary = token_dictionary
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.transformers_count = transformers_count
        self.ff_network_size = ff_network_size

        pass

class Model(nn.Module):
    def __init__(self, token_dictionary: TokenDictionary, device: device, vocabulary_size: int, embedding_size: int, context_size: int, transformers_count: int, ff_network_size: int) -> None:
        super().__init__()
        
        self.token_dictionary = token_dictionary
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.transformers_count = transformers_count
        self.ff_network_size = ff_network_size

        self.embedding_layer = EmbeddingLayer(vocabulary_size, embedding_size)
        self.position_encoding_layer = PositionEncodingLayer(context_size, embedding_size)
        self.transformers = nn.ModuleList([
            TransformerLayer(ff_network_size, context_size, embedding_size) for _ in range(transformers_count)
        ])
        self.output_layer = OutputLayer(vocabulary_size, embedding_size)
        self.to(device)

    def inference(self, token_ids: list[int], candidates: int) -> list[tuple[int, float]]:
        input_tensor = torch.tensor(token_ids).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        output_tensor = self(input_tensor)

        output_vector = output_tensor[:, -1, :]
        probability_vector = torch.softmax(output_vector, dim=-1)
        probability_vector_sorted = torch.sort(probability_vector, descending=True)
        output = []

        for i in range(candidates):
            token_id = int(probability_vector_sorted.indices[0][i].item())
            probability = float(probability_vector[0][token_id].item())
            output.append((token_id, probability))

        return output
    
    def forward(self, x : Tensor) -> Tensor:
        x = self.embedding_layer(x)
        x = self.position_encoding_layer(x)

        for transformer in self.transformers:
            x = transformer(x)
        x = self.output_layer(x)

        return x
    
    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def load(path: str, device_name: str):
        persistence = torch.load(path, weights_only=False)
        model = Model(
            persistence.token_dictionary,
            device=torch.device(device_name),
            vocabulary_size=persistence.vocabulary_size,
            embedding_size=persistence.embedding_size,
            context_size=persistence.context_size,
            transformers_count=persistence.transformers_count, 
            ff_network_size=persistence.ff_network_size
        )
        model.load_state_dict(persistence.state)

        return model

    def save(self, path: str):
        persistence = ModelPersistence(
            self.state_dict(),
            self.token_dictionary,
            self.vocabulary_size,
            self.embedding_size,
            self.context_size,
            self.transformers_count,
            self.ff_network_size
        )
        torch.save(persistence, path)
    
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

        self.register_buffer("sequence", torch.arange(context_size))

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        sequence = self.sequence[:x.size(1)] # pyright: ignore[reportIndexIssue]
        position_encoding = self.position_matrix(sequence)

        # [0.1; 0.2; 0.3]   [0.1; 0.1; 0.1]   [0.2; 0.3; 0.4]
        # [0.2; 0.3; 0.4] + [0.2; 0.2; 0.2] = [0.4; 0.5; 0.6]
        # [0.3; 0.4; 0.5]   [0.3; 0.3; 0.3]   [0.6; 0.7; 0.8]
        return x + position_encoding.unsqueeze(0)
    
class TransformerLayer(nn.Module):
    def __init__(self, ff_network_size: int, context_size: int, embedding_size: int) -> None:
        super().__init__()

        self.attention_layer = SelfAttentionLayer(context_size, embedding_size)
        self.attention_norm = nn.LayerNorm(embedding_size)

        self.ff_network_layer = FeedForwardNetworkLayer(ff_network_size, embedding_size)
        self.ff_network_norm = nn.LayerNorm(embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        # Pre-LN
        x = x + self.attention_layer(self.attention_norm(x))
        x = x + self.ff_network_layer(self.ff_network_norm(x))

        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, context_size: int, embedding_size: int) -> None:
        super().__init__()
        
        self.context_size = context_size
        self.embedding_size = embedding_size

        # Three matrices (unified into one) to represent Query, Key and Value
        # Query - what tokens are looking for
        # Key - what tokens are representing
        # Value - what tokens are providing
        self.qkv_matrix = nn.Linear(embedding_size, embedding_size * 3)

        self.register_buffer("mask", torch.triu(torch.ones(context_size, context_size, dtype=torch.bool), diagonal=1).unsqueeze(0))

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        # Embedded and position encoded tokens are projected into Q, K and V matrices
        qkv = self.qkv_matrix(x) # [batch; sequence_size; embedding_size * 3]
        q, k, v = qkv.chunk(3, dim=-1) # [batch; sequence_size; embedding_size]

        # Calculate attention weights by multiplying Query and Key
        attention_weights = q @ torch.transpose(k, -2, -1) # [batch; sequence_size; sequence_size]

        # Apply a precaculated mask, so token can see only previous attention scores
        mask = self.mask[:, :x.size(1), :x.size(1)] # pyright: ignore[reportIndexIssue]
        attention_weights = attention_weights.masked_fill(mask, float("-inf"))

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
    
class OutputLayer(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.output_matrix = nn.Linear(embedding_size, vocabulary_size, bias=False)
        self.output_norm = nn.LayerNorm(embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; vocabulary_size]

        #return self.output_matrix(self.output_norm(x))
        return self.output_matrix(x)
