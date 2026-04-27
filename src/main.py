import argparse

from model import Model
from training import Trainer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-t", "--training", action="store_true")
    args = parser.parse_args()

    if args.inference:
        entry_point_inference()
    elif args.training:
        entry_point_training()
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

def entry_point_training() -> None:
    pass

if __name__ == "__main__":
    main()