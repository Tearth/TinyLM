import model;

from model import Model;

def main():
    model = Model()

    input = "floppa"
    output = model.prompt(input)

    print(" > " + input)
    print(" < " + output)

if __name__ == "__main__":
    main()