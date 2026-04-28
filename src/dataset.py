import torch

from tokens import TokenDictionary
from torch import Tensor
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, token_dictionary: TokenDictionary, chunk_size: int) -> None:
        self.token_dictionary = token_dictionary
        self.chunk_size = chunk_size
        self.chunks = []
        self.size = 0

    def load(self, path: str):
        pad_id = self.token_dictionary.encode_pad()

        with open(path, encoding="utf-8") as file:
            for line in file:
                _, token_ids = self.token_dictionary.encode(line, False)
                
                start_position = 0
                self.size += len(line)

                while start_position + self.chunk_size <= len(token_ids):
                    self.chunks.append(token_ids[start_position:start_position + self.chunk_size])
                    start_position += self.chunk_size
                
                if start_position < len(token_ids):
                    chunk = token_ids[start_position:]
                    chunk.extend([pad_id] * (self.chunk_size - (len(token_ids) - start_position)))
                    self.chunks.append(chunk)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        chunk = self.chunks[idx]
        features = chunk[:]
        labels = chunk[1:] + [self.token_dictionary.encode_pad()]

        return torch.tensor(features), torch.tensor(labels)

    def __len__(self):
        return len(self.chunks)