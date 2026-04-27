from tokens import TokenDictionary

class Dataset:
    def __init__(self, token_dictionary: TokenDictionary, chunk_size: int) -> None:
        self.token_dictionary = token_dictionary
        self.chunk_size = chunk_size
        self.chunks = []

    def load(self, path: str):
        pad_id = self.token_dictionary.encode_pad()

        with open(path) as file:
            for line in file:
                _, token_ids = self.token_dictionary.encode(line, False)
                start_position = 0

                while start_position + self.chunk_size <= len(token_ids):
                    self.chunks.append(token_ids[start_position:start_position + self.chunk_size])
                    start_position += self.chunk_size
                
                if start_position < len(token_ids):
                    chunk = token_ids[start_position:]
                    chunk.extend([pad_id] * (self.chunk_size - (len(token_ids) - start_position)))
                    self.chunks.append(chunk)

