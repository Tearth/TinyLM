from model import Model;

def main():
    vocabulary_size = 3
    embedding_size = 4
    context_size = 16
    transformers_count = 2
    ff_network_size = 8

    model = Model(vocabulary_size, embedding_size, context_size, transformers_count, ff_network_size)

    input = "Floppa1 Floppa2 Floppa3"
    output = model.prompt(input)

    print("---")
    print(" > " + input)
    print(" < " + output)

if __name__ == "__main__":
    main()