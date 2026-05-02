import torch

from torch import device, nn
from torch import Tensor
from tokens import TokenDictionary
from typing import Any, Self

class ModelPersistence:
    def __init__(
        self, state: dict[str, Any],
        token_dictionary: TokenDictionary,
        vocabulary_size: int,
        embedding_size: int,
        context_size: int,
        transformers_count: int,
        heads_count: int,
        ff_network_size: int,
        dropout_rate: float
    ) -> None:
        self.state = state
        self.token_dictionary = token_dictionary
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.transformers_count = transformers_count
        self.heads_count = heads_count
        self.ff_network_size = ff_network_size
        self.dropout_rate = dropout_rate

class Model(nn.Module):
    def __init__(
        self, 
        token_dictionary: TokenDictionary, 
        device: device, 
        vocabulary_size: int, 
        embedding_size: int, 
        context_size: int, 
        transformers_count: int, 
        heads_count: int, 
        ff_network_size: int,
        dropout_rate: float
    ) -> None:
        super().__init__()
        
        self.token_dictionary = token_dictionary
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.transformers_count = transformers_count
        self.heads_count = heads_count
        self.ff_network_size = ff_network_size
        self.dropout_rate = dropout_rate

        self.embedding_layer = EmbeddingLayer(vocabulary_size, embedding_size)
        self.position_encoding_layer = PositionEncodingLayer(context_size, embedding_size, self.dropout_rate)
        self.transformers = nn.ModuleList([
            TransformerLayer(heads_count, ff_network_size, context_size, embedding_size, self.dropout_rate) for _ in range(transformers_count)
        ])
        self.output_layer = OutputLayer(vocabulary_size, embedding_size)

    def predict(self, token_ids: list[int], candidates: int) -> list[tuple[int, float]]:
        # Prepare input tensor as a single batch
        input_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(0)

        # Run inference and extract the last vector, representing a next token probabilities
        output_tensor = self(input_tensor)
        output_vector = output_tensor[:, -1, :]

        # Normalize probabilities and sort them
        probability_vector = torch.softmax(output_vector, dim=-1)
        probability_vector_sorted = torch.topk(probability_vector, k=candidates)
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

    @classmethod
    def load(cls, path: str, device_name: str) -> Self:
        persistence = torch.load(path, weights_only=False)
        model = cls(
            persistence.token_dictionary,
            device=torch.device(device_name),
            vocabulary_size=persistence.vocabulary_size,
            embedding_size=persistence.embedding_size,
            context_size=persistence.context_size,
            transformers_count=persistence.transformers_count,
            heads_count = persistence.heads_count,
            ff_network_size=persistence.ff_network_size,
            dropout_rate=persistence.dropout_rate
        )
        model.load_state_dict(persistence.state)

        return model

    def save(self, path: str) -> None:
        persistence = ModelPersistence(
            self.state_dict(),
            self.token_dictionary,
            self.vocabulary_size,
            self.embedding_size,
            self.context_size,
            self.transformers_count,
            self.heads_count,
            self.ff_network_size,
            self.dropout_rate
        )
        torch.save(persistence, path)
    
class EmbeddingLayer(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int) -> None:
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Embedding with each row representing a vector for a particular token
        self.embedding_matrix = nn.Embedding(vocabulary_size, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        # Input: [batch; sequence_size]
        # Output: [batch; sequence_size; embedding_size]
        
        # Look-up vectors for each token in input
        return self.embedding_matrix(x)
    
class PositionEncodingLayer(nn.Module):
    def __init__(self, context_size: int, embedding_size: int, dropout_rate: float) -> None:
        super().__init__()
        
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        # Embedding with each row representing a vector for the position in the context
        self.position_matrix = nn.Embedding(context_size, embedding_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)

        # Static sequence
        # [0, 1, ..., context_size]
        self.register_buffer("sequence",
            torch.arange(context_size)
        )

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        # Adjust static sequence size to input, so it's possible to add both to each other
        sequence = self.sequence[:x.size(1)] # pyright: ignore[reportIndexIssue]

        # Get position vectors for every input token 
        position_encoding = self.position_matrix(sequence)

        # Sum embeddings and unique position vectors
        x = x + position_encoding.unsqueeze(0)
        x = self.output_dropout(x)

        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, heads_count: int, ff_network_size: int, context_size: int, embedding_size: int, dropout_rate: float) -> None:
        super().__init__()

        # Self-attention layer with attention mask to ensure that token can see only previous ones
        self.attention_layer = SelfAttentionLayer(heads_count, context_size, embedding_size, dropout_rate)
        self.attention_norm = nn.LayerNorm(embedding_size)

        # Feed-forward network for additional input processing
        self.ff_network_layer = FeedForwardNetworkLayer(ff_network_size, embedding_size, dropout_rate)
        self.ff_network_norm = nn.LayerNorm(embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        # Apply self-attention with Pre-LN normalization to stabilize training
        residual = x
        x = self.attention_norm(x)
        x = self.attention_layer(x)
        x = residual + x

        # Apply feed-forward network with Pre-LN normalization to stabilize training
        residual = x
        x = self.ff_network_norm(x)
        x = self.ff_network_layer(x)
        x = residual + x

        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, heads_count: int, context_size: int, embedding_size: int, dropout_rate: float) -> None:
        super().__init__()
        
        self.heads_count = heads_count
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.embedding_size_per_head = embedding_size // heads_count
        self.dropout_rate = dropout_rate

        # Query-Key-Value matrices unified into one big matrix to reduce separate multiplications
        self.qkv_matrix = nn.Linear(embedding_size, embedding_size * 3)
        self.attention_dropout = nn.Dropout(self.dropout_rate)

        # Output matrix projecting attention value into final result
        self.output_matrix = nn.Linear(embedding_size, embedding_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)

        # Static mask
        # [ 0, 1, 1 ]
        # [ 0, 0, 1 ]
        # [ 0, 0, 0 ]
        self.register_buffer("mask",
            torch.triu(torch.ones(context_size, context_size, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        # Save input dimensions for later use
        B, S, _ = x.size()

        # Embedded tokens are projected into Q, K and V matrices
        # Query - what tokens are looking for
        # Key - what tokens are representing
        # Value - what tokens are providing
        qkv = self.qkv_matrix(x) # [batch; sequence_size; embedding_size * 3]

        # Split qkv into separate matrices, representing Query, Key or Value
        q, k, v = qkv.chunk(3, dim=-1) # [batch; sequence_size; embedding_size]

        # Project matrices into [batch; sequence_size; heads_count; embeddings_size_per_head] and swap sequence_size with heads_count
        q = q.view(B, S, self.heads_count, self.embedding_size_per_head).transpose(1, 2) # [batch; heads_count; sequence_size; embeddings_size_per_head]
        k = k.view(B, S, self.heads_count, self.embedding_size_per_head).transpose(1, 2) # [batch; heads_count; sequence_size; embeddings_size_per_head]
        v = v.view(B, S, self.heads_count, self.embedding_size_per_head).transpose(1, 2) # [batch; heads_count; sequence_size; embeddings_size_per_head]

        # Reshape precalculated mask to make it compatible with attention_weights
        mask = self.mask[None, None, :S, :S] # [1; 1; sequence_size; sequence_size] # type: ignore

        # Calculate attention weights by multiplying Query and transposed Key
        attention_weights = q @ torch.transpose(k, 2, 3) # [batch; heads_count; sequence_size; sequence_size]

        # Apply a precaculated mask, so token can see only preceding attention values
        attention_weights = attention_weights.masked_fill(mask, float("-inf")) # [batch; heads_count; sequence_size; sequence_size]

        # Scale and normalize attention weights, so the sum of probability is equal to 1.0
        attention_weights = torch.softmax(attention_weights / (self.embedding_size_per_head ** 0.5), dim=-1) # [batch; heads_count; sequence_size; sequence_size]
        attention_weights = self.attention_dropout(attention_weights) # [batch; heads_count; sequence_size; sequence_size]

        # Multiply attention weights with Value to get the final values according to their relevance
        attention_value = attention_weights @ v # [batch; heads_count; sequence_size; embedding_size]
        
        # Swap sequence_size with heads_count again, so heads can be merged 
        attention_value = torch.transpose(attention_value, 1, 2) # [batch; sequence_size; heads_count; embedding_size]

        # Reshape attention_value to [batch; sequence_size; embedding_size]
        attention_value = attention_value.contiguous().view(B, S, self.embedding_size) # [batch; sequence_size; embedding_size]

        # Process all separately calculated attention values into coherent output
        output = self.output_matrix(attention_value)
        output = self.output_dropout(output)

        return output
    
class FeedForwardNetworkLayer(nn.Module):
    def __init__(self, ff_network_size: int, embedding_size: int, dropout_rate: float) -> None:
        super().__init__()

        self.ff_network_size = ff_network_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        # Two linear layers with ReLu activation function to process self-attention output
        self.layer_a = nn.Linear(embedding_size, ff_network_size)
        self.layer_b = nn.Linear(ff_network_size, embedding_size)
        self.layer_a_dropout = nn.Dropout(dropout_rate)
        self.layer_b_dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
    
    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; embedding_size]

        x = self.layer_a(x) # [batch; sequence_size; ff_network_size]
        x = self.activation(x) # [batch; sequence_size; ff_network_size]
        x = self.layer_a_dropout(x) # [batch; sequence_size; ff_network_size]
        x = self.layer_b(x) # [batch; sequence_size; ff_network_size]
        x = self.layer_b_dropout(x) # [batch; sequence_size; ff_network_size]

        return x
    
class OutputLayer(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int) -> None:
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Projection from embeddings to logits
        self.output_matrix = nn.Linear(embedding_size, vocabulary_size, bias=False)

    def forward(self, x : Tensor) -> Tensor:
        # Input: [batch; sequence_size; embedding_size]
        # Output: [batch; sequence_size; vocabulary_size]

        return self.output_matrix(x)
