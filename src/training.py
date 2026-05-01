import logging
import time
import torch

from torch import GradScaler, Tensor, autocast, nn
from model import Model
from dataset import Dataset
from tokens import TokenDictionary
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model: Model,
        output_path: str,
        dataset: Dataset, 
        token_dictionary: TokenDictionary,
        max_epoch: int,
        batch_size: int, 
        learning_rate: float, 
        beta1: float, 
        beta2: float, 
        weight_decay: float,
        save_interval: int
    ) -> None:
        self.model = model
        self.output_path = output_path
        self.dataset = dataset
        self.token_dictionary = token_dictionary
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.save_interval = save_interval
        
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
        scaler = GradScaler()
        for epoch in range(1, self.max_epoch + 1):
            total_loss = 0
            batches = 0
            timestamp = time.time()

            for (features_batch, labels_batch) in self.dataloader:
                features_batch = features_batch.to(self.model.device)
                labels_batch = labels_batch.to(self.model.device)

                with autocast(device_type="cuda"):
                    forward_pass_outputs = self.model(features_batch)

                    # Model output and labels must be flattened first, so there's not explicit batch dimension
                    loss = self.loss_function(
                        forward_pass_outputs.view(forward_pass_outputs.size(dim=0) * forward_pass_outputs.size(dim=1), forward_pass_outputs.size(dim=2)), 
                        labels_batch.view(labels_batch.size(dim=0) * labels_batch.size(dim=1))
                    )
                
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()
                batches += 1

            average_loss = total_loss / batches
            delta_time = time.time() - timestamp

            if (epoch % self.save_interval) == 0:
                self.model.save(self.output_path)

            logging.info(f"Epoch {epoch} done in {delta_time:.2f} seconds, average loss: {average_loss:.4f}")