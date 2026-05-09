import logging
import os
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
        read_bytes = 0
        read_bytes_total = 0
        file_size = os.path.getsize(path)

        with open(path, encoding="utf-8") as file:
            for line in file:
                buffer.extend(self.token_dictionary.encode_line(line))
                read_bytes += len(line.encode())

                if read_bytes >= file_size / 10:
                    read_bytes_total += read_bytes
                    read_bytes = 0
                    percent = read_bytes_total * 100 // file_size

                    logging.debug(f"Read {percent}% ({read_bytes_total}/{file_size} bytes)")

        self.data = torch.tensor(buffer)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        position = idx * self.stride
        features = self.data[position:position+self.chunk_size]
        labels = self.data[position + 1:position+self.chunk_size + 1]

        return (features, labels)

    def __len__(self):
        return int(len(self.data) - self.chunk_size) // self.stride