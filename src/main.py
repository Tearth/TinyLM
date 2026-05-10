import argparse
import logging
import os
import random
import time
import torch

from model import Model
from training import Trainer
from dataset import ModelDataset


def main() -> None:
    parser = argparse.ArgumentParser()

    inference_group = parser.add_mutually_exclusive_group(required=True)
    inference_group.add_argument("-i", "--inference", action="store_true", help="select inference mode")
    inference_group.add_argument("-t", "--training", action="store_true", help="select training mode")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("-c", "--cpu", action="store_true", help="set training to use CPU only")
    device_group.add_argument("-g", "--gpu", action="store_true", help="set training to use GPU via CUDA")
    parser.add_argument("-m", "--model", help="path to the model (works both in inference and training mode)")
    parser.add_argument("-d", "--dataset", help="path to the text file (works only in training mode)")
    parser.add_argument("-o", "--output", help="path to the output model (works in training mode only)")
    parser.add_argument("-p", "--prompt", help="prompt for inference mode")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s | %(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.inference and not args.model:
        parser.error("--model is required for inference mode")

    if args.inference and not args.prompt:
        parser.error("--prompt is required for inference mode")

    if args.training and not args.dataset:
        parser.error("--dataset is required for training mode")

    if args.training and not args.output:
        parser.error("--output is required for training mode")

    if args.cpu:
        device_name = "cpu"
    elif args.gpu:
        device_name = "cuda"
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if args.inference:
        entry_point_inference(args.model, args.prompt, device_name)
    elif args.training:
        entry_point_training(args.model, args.dataset, args.output, device_name)


def entry_point_inference(model_path: str, prompt: str, device_name: str) -> None:
    logging.info(f"========== INFERENCE MODE ==========")
    logging.info(f"Model: {model_path}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Device: {device_name}")
    logging.info(f"====================================")
    logging.info(f"Loading model...")
    torch.no_grad()

    timestamp = time.time()
    model = Model.load(model_path, device_name)
    model.to(model.device)
    model.eval()
    delta_time = time.time() - timestamp

    input = prompt

    logging.info(f"Done in {delta_time:.2f} seconds")
    logging.info(f"Model:")
    logging.info(f"- parameters: {model.parameters_count()}")
    logging.info(f"- embedding size: {model.embedding_size}")
    logging.info(f"- context size: {model.context_size}")
    logging.info(f"- transformers: {model.transformers_count}")
    logging.info(f"- heads count: {model.heads_count}")
    logging.info(f"- ff network size: {model.ff_network_size}")
    logging.info(f"- dropout rate: {model.dropout_rate}")
    logging.info(f"Output:")

    print("-------------------------------------")
    print(prompt, end="")

    for _ in range(2048):
        context = input[-model.context_size :]
        context_token_ids = model.token_dictionary.encode_block(context)

        candidates = model.predict(context_token_ids, 4)

        keys = list(map(lambda x: x[0], candidates))
        probabilities = list(map(lambda x: x[1], candidates))
        next_token_id = random.choices(keys, probabilities)[0]
        next_token = model.token_dictionary.decode_token(next_token_id)

        input = input + next_token
        print(next_token, end="")


def entry_point_training(model_path: str | None, dataset_path: str, output_path: str, device_name: str) -> None:
    logging.info(f"========== TRAINING MODE ==========")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Device: {device_name}")
    logging.info(f"===================================")

    # fmt: off
    dataset = ModelDataset(
        chunk_size=256,
        stride=128
    )
    dataset_binary_path = dataset_path + ".bin"

    if os.path.isfile(dataset_binary_path):
        logging.info(f"Loading dataset (found a binary)...")

        timestamp = time.time()
        dataset.load_bin(dataset_binary_path)
        delta_time = time.time() - timestamp

        logging.info(f"Done in {delta_time:.2f} seconds, loaded {len(dataset.data)} tokens")
    else:
        logging.info(f"Building token dictionary...")

        timestamp = time.time()
        dataset.token_dictionary.build(dataset_path, 3000)
        delta_time = time.time() - timestamp

        logging.info(f"Done in {delta_time:.2f} seconds, constructed {len(dataset.token_dictionary.map)} tokens")
        logging.info(f"Loading dataset and saving to {dataset_binary_path}...")

        timestamp = time.time()
        dataset.load_txt(dataset_path)
        dataset.save_bin(dataset_binary_path)
        delta_time = time.time() - timestamp

        logging.info(f"Done in {delta_time:.2f} seconds, loaded {len(dataset.data)} tokens")
    
    logging.info(f"Dataset:")
    logging.info(f"- chunks: {len(dataset)}")
    logging.info(f"- chunk size: {dataset.chunk_size}")

    if model_path is None:
        logging.info(f"Loading model...")

        timestamp = time.time()
        model = Model(
            dataset.token_dictionary,
            device=torch.device(device_name),
            vocabulary_size=len(dataset.token_dictionary.map),
            embedding_size=192,
            context_size=256,
            transformers_count=6,
            heads_count=4,
            ff_network_size=768,
            dropout_rate=0.1,
        )
        delta_time = time.time() - timestamp

        logging.info(f"Done in {delta_time:.2f} seconds")
    else:
        model = Model.load(model_path, device_name)

    logging.info(f"Compiling model...")

    timestamp = time.time()
    model.to(model.device)
    model.train()
    model.compile()
    delta_time = time.time() - timestamp

    logging.info(f"Done in {delta_time:.2f} seconds")
    logging.info(f"Model:")
    logging.info(f"- parameters: {model.parameters_count()}")
    logging.info(f"- embedding size: {model.embedding_size}")
    logging.info(f"- context size: {model.context_size}")
    logging.info(f"- transformers: {model.transformers_count}")
    logging.info(f"- heads count: {model.heads_count}")
    logging.info(f"- ff network size: {model.ff_network_size}")
    logging.info(f"- dropout rate: {model.dropout_rate}")

    trainer = Trainer(
        model,
        output_path,
        dataset,
        max_epoch=10000,
        batch_size=64,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.01,
        save_interval=1,
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
