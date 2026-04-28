import logging
import time
import torch

from torch import Tensor, nn
from model import Model
from dataset import Dataset
from tokens import TokenDictionary
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self, model: Model, 
        dataset: Dataset, 
        token_dictionary: TokenDictionary,
        max_epoch: int,
        batch_size: int, 
        learning_rate: float, 
        beta1: float, 
        beta2: float, 
        weight_decay: float
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.token_dictionary = token_dictionary
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )

        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.token_dictionary.encode_pad()
        )

    def run(self) -> None:
        for epoch in range(1, self.max_epoch + 1):
            total_loss = 0
            batches = 0
            timestamp = time.time()

            for (features_batch, labels_batch) in self.dataloader:
                forward_pass_outputs = self.model(features_batch)

                # Model output and labels must be flattened first, to get rid of batch dimension
                loss = self.loss_function(
                    forward_pass_outputs.view(forward_pass_outputs.size(dim=0) * forward_pass_outputs.size(dim=1), forward_pass_outputs.size(dim=2)), 
                    labels_batch.view(labels_batch.size(dim=0) * labels_batch.size(dim=1))
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            average_loss = total_loss / batches
            delta_time = time.time() - timestamp

            logging.info(f"Epoch {epoch} done in {delta_time:.2f} seconds, average loss: {average_loss:.4f}")