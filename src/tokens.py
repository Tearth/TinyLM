
from bidict import bidict

UNKNOWN_TOKEN = "<unknown>"
PAD_TOKEN = "<pad>"

class TokenDictionary:
    def __init__(self) -> None:
        self.map = bidict({
            UNKNOWN_TOKEN: 0,
            PAD_TOKEN: 1,
        })
        self.next_id = 2

    def encode(self, content: str, read_only: bool) -> tuple[list[str], list[int]]:
        tokens = content.split()
        token_ids = []

        for token in tokens:
            if read_only:
                token_ids.append(self.map.get(token, self.map[UNKNOWN_TOKEN]))
            else:
                id = self.map.get(token)
                
                if id is None:
                    id = self.next_id
                    self.map.put(token, self.next_id)
                    self.next_id += 1
                
                token_ids.append(id)

        return (tokens, token_ids)

    def encode_pad(self) -> int:
        return self.map[PAD_TOKEN]
    
    def decode(self, id: int) -> str:
        return self.map.inverse[id]