import argparse
import logging
import random
import torch

from model import Model
from training import Trainer
from dataset import ModelDataset
from tokens import TokenDictionary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-t", "--training", action="store_true")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument("-g", "--gpu", action="store_true")
    parser.add_argument("-m", "--model")
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-o", "--output")
    parser.add_argument("-p", "--prompt")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-7s | %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.cpu:
        device_name = "cpu"
    elif args.gpu:
        device_name = "cuda"
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if args.inference:
        entry_point_inference(args.model, args.prompt, device_name)
    elif args.training:
        entry_point_training(args.dataset, args.output, device_name)
    else:
        logging.error("No mode selected")

def entry_point_inference(model_path: str, prompt: str, device_name: str) -> None:
    logging.info(f"========== INFERENCE MODE ==========")
    logging.info(f"Model: {model_path}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Device: {device_name}")
    logging.info(f"====================================")
    logging.info(f"Loading model...")
    
    model = Model.load(model_path, device_name)
    
    logging.info(f"Model:")
    logging.info(f"- parameters: {model.parameters_count()}")
    logging.info(f"- embedding size: {model.embedding_size}")
    logging.info(f"- context size: {model.context_size}")
    logging.info(f"- transformers count: {model.transformers_count}")
    logging.info(f"- ff network size: {model.ff_network_size}")
    logging.info(f"Output:")
    print(prompt, end="")

    input = prompt
    
    for _ in range(2038):
        context = input[-model.context_size:]
        context_token_ids = []

        for char in context:
            context_token_ids.append(model.token_dictionary.encode(char, True))
        
        candidates = model.inference(context_token_ids, 8)

        keys = list(map(lambda x: x[0], candidates))
        probabilities = list(map(lambda x: x[1], candidates))
        next_token_id = random.choices(keys, probabilities)[0]
        next_token = model.token_dictionary.decode(next_token_id)

        input = input + next_token
        print(next_token, end="")
    pass

def entry_point_training(dataset_path: str, output_path: str, device_name: str) -> None:
    logging.info(f"========== TRAINING MODE ==========")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Device: {device_name}")
    logging.info(f"===================================")
    logging.info(f"Loading dataset...")

    token_dictionary = TokenDictionary()
    dataset = ModelDataset(
        token_dictionary, 
        chunk_size=128,
        stride=32
    )
    dataset.load(dataset_path)

    logging.info(f"Done, loaded {dataset.size} bytes ({dataset.size / 1024 / 1024:.2f} MB)")
    logging.info(f"Dataset :")
    logging.info(f"- vocabulary size: {len(dataset.token_dictionary.map)}")
    logging.info(f"- chunks: {len(dataset)}")
    logging.info(f"- chunk size: {dataset.chunk_size}")

    model = Model(
        token_dictionary,
        device=torch.device(device_name),
        vocabulary_size=len(token_dictionary.map),
        embedding_size=64,
        context_size=128,
        transformers_count=4, 
        ff_network_size=256
    )
    model.train()

    logging.info(f"Model:")
    logging.info(f"- parameters: {model.parameters_count()}")
    logging.info(f"- embedding size: {model.embedding_size}")
    logging.info(f"- context size: {model.context_size}")
    logging.info(f"- transformers: {model.transformers_count}")
    logging.info(f"- ff network size: {model.ff_network_size}")
    
    trainer = Trainer(
        model,
        output_path,
        dataset,
        token_dictionary,
        max_epoch=10000,
        batch_size=256,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.01,
        save_interval=1
    )

    logging.info(f"Trainer:")
    logging.info(f"- max epoch: {trainer.max_epoch}")
    logging.info(f"- batch size: {trainer.batch_size}")
    logging.info(f"- learning rate: {trainer.learning_rate}")
    logging.info(f"- beta1: {trainer.beta1}")
    logging.info(f"- beta2: {trainer.beta2}")
    logging.info(f"- weight decay: {trainer.weight_decay}")
    logging.info(f"Switching to training loop")
 
    trainer.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Unexpected error: {ex}")
        raise