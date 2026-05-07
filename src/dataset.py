import torch

from tokens import TokenDictionary
from torch import Tensor
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, token_dictionary: TokenDictionary, chunk_size: int, stride: int) -> None:
        self.token_dictionary = token_dictionary
        self.chunk_size = chunk_size
        self.stride = stride
        self.data = torch.empty(0)

    def load(self, path: str):
        buffer = []

        with open(path, encoding="utf-8") as file:
            for line in file:
                buffer.extend(self.token_dictionary.encode_block(line))

        self.data = torch.tensor(buffer)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        position = int(idx * self.stride)
        features = self.data[position:position+self.chunk_size]
        labels = self.data[position + 1:position+self.chunk_size + 1]

        return (features, labels)

    def __len__(self):
        return int(len(self.data) - self.chunk_size) // self.stride