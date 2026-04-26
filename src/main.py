from model import Model

def main():
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

if __name__ == "__main__":
    main()