import logging
import os
import torch

from tokens import DOCUMENT_END, TokenDictionary
from torch import Tensor
from torch.utils.data import Dataset


class ModelDatasetPersistence:
    def __init__(self, data: Tensor, document_ids: Tensor, token_dictionary: TokenDictionary) -> None:
        self.data = data
        self.document_ids = document_ids
        self.token_dictionary = token_dictionary


class ModelDataset(Dataset):
    def __init__(self, chunk_size: int, stride: int) -> None:
        self.token_dictionary = TokenDictionary()
        self.chunk_size = chunk_size
        self.stride = stride
        self.data = torch.empty(0)
        self.document_ids = torch.empty(0)

    def load_txt(self, path: str):
        buffer = []
        document_ids = []

        read_bytes = 0
        read_bytes_total = 0
        file_size = os.path.getsize(path)
        document_id = 0

        end_document_token_id = self.token_dictionary.encode_token(DOCUMENT_END)[0]

        with open(path, encoding="utf-8") as file:
            for line in file:
                encoded_line = self.token_dictionary.encode_line(line)
                buffer.extend(encoded_line)
                read_bytes += len(line.encode())

                for token in encoded_line:
                    document_ids.append(document_id)
                    if token == end_document_token_id:
                        document_id += 1

                if read_bytes >= file_size / 10:
                    read_bytes_total += read_bytes
                    read_bytes = 0
                    percent = read_bytes_total * 100 // file_size

                    logging.debug(f"Read {percent}% ({read_bytes_total}/{file_size} bytes)")

        self.data = torch.tensor(buffer)
        self.document_ids = torch.tensor(document_ids)

    def load_bin(self, path: str):
        persistence = torch.load(path, weights_only=False)
        self.data = persistence.data
        self.document_ids = persistence.document_ids
        self.token_dictionary = persistence.token_dictionary

    def save_bin(self, path: str):
        persistence = ModelDatasetPersistence(
            self.data,
            self.document_ids,
            self.token_dictionary,
        )
        torch.save(persistence, path)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        position = idx * self.stride
        features = self.data[position : position + self.chunk_size]
        labels = self.data[position + 1 : position + self.chunk_size + 1]
        document_ids = self.document_ids[position : position + self.chunk_size]

        return (features, labels, document_ids)

    def __len__(self):
        return int(len(self.data) - self.chunk_size) // self.stride
