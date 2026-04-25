import torch
import bidict

from torch import nn;
from torch import Tensor;
from bidict import bidict;

class Model:
    def __init__(self, vocabulary_size: int, embedding_size: int, context_size: int):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.tokenizer = Tokenizer()
        self.encoder_dekoder = EncoderDecoder()
        self.embedding_layer = EmbeddingLayer(vocabulary_size, embedding_size)
        self.position_encoding_layer = PositionEncodingLayer(context_size, embedding_size)

    def prompt(self, input: str) -> str:
        print("Input: " + input)

        # Split input into a list of separate tokens, where each one represents a single lower-cased word - no interpunction supported for now
        tokens = self.tokenizer.get_tokens(input)
        token_ids = self.encoder_dekoder.encode_list(tokens)

        print("Tokens: " + ", ".join(tokens))
        print("Token IDs: " + ", ".join(map(str, token_ids)))

        return "capybara"
    
class Tokenizer:
    def __init__(self):
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
    
    def encode_list(self, list) -> list[int]:
        result = []

        for token in list:
            result.append(self.encode(token))
        
        return result
    
    def decode(self, id) -> str:
        return self.map.inverse[id]
    
class EmbeddingLayer(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        # Each token has unique vector of `embedding_size` length
        self.embedding_matrix = nn.Embedding(vocabulary_size, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        # [ 3, 1, 2]
        #
        # becomes
        #
        # [0.1; 0.2; 0.3]
        # [0.2; 0.3; 0.4]
        # [0.3; 0.4; 0.5]
        return self.embedding_matrix(x)
    
class PositionEncodingLayer(nn.Module):
    def __init__(self, context_size: int, embedding_size: int):
        super().__init__()
        
        self.context_size = context_size
        self.embedding_size = embedding_size

        # We position in context has unique vector of `embedding_size` length
        self.position_matrix = nn.Embedding(context_size, embedding_size)

    def forward(self, x : Tensor) -> Tensor:
        sequence = torch.arange(0, x.size(0), device=x.device)
        position_encoding = self.position_matrix(sequence)

        # [0.1; 0.2; 0.3]   [0.1; 0.1; 0.1]   [0.2; 0.3; 0.4]
        # [0.2; 0.3; 0.4] + [0.2; 0.2; 0.2] = [0.4; 0.5; 0.6]
        # [0.3; 0.4; 0.5]   [0.3; 0.3; 0.3]   [0.6; 0.7; 0.8]
        return x + position_encoding