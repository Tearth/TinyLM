import argparse
import logging

from model import Model
from training import Trainer
from dataset import ModelDataset
from tokens import TokenDictionary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-t", "--training", action="store_true")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--dataset")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-7s | %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.inference:
        entry_point_inference()
    elif args.training:
        entry_point_training(args.model, args.dataset)
    else:
        print("No mode selected")

def entry_point_inference() -> None:
    model = Model(
        vocabulary_size=3,
        embedding_size=4,
        context_size=16,
        transformers_count=2, 
        ff_network_size=8
    )
    input = "Floppa1 Floppa2 Floppa3"

    print(" < " + input)
    
    for _ in range(5):
        output = model.prompt(input)
        input = input + " " + output
        print(" < " + input)

def entry_point_training(model_path: str, dataset_path: str) -> None:
    logging.info(f"========== TRAINING MODE ==========")
    logging.info(f"Model: {model_path}")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"===================================")
    logging.info(f"Loading dataset...")

    token_dictionary = TokenDictionary()
    dataset = ModelDataset(token_dictionary, chunk_size=16)
    dataset.load(dataset_path)

    logging.info(f"Done, loaded {dataset.size} bytes ({dataset.size / 1024 / 1024:.2f} MB)")
    logging.info(f"Dataset parameters:")
    logging.info(f"- vocabulary size: {len(dataset.token_dictionary.map)}")
    logging.info(f"- chunks count: {len(dataset.chunks)}")
    logging.info(f"- chunk size: {dataset.chunk_size}")

    model = Model(
        vocabulary_size=len(token_dictionary.map),
        embedding_size=16,
        context_size=16,
        transformers_count=2, 
        ff_network_size=8
    )
    model.train()

    logging.info(f"Model parameters:")
    logging.info(f"- embedding size: {model.embedding_size}")
    logging.info(f"- context size: {model.context_size}")
    logging.info(f"- transformers count: {model.transformers_count}")
    logging.info(f"- ff network size: {model.ff_network_size}")
    
    trainer = Trainer(
        model, 
        dataset, 
        token_dictionary,
        max_epoch = 10000,
        batch_size=64,
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1
    )

    logging.info(f"Trainer parameters:")
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