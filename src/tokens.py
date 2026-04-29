from bidict import bidict

UNKNOWN_TOKEN = "⊗"
PAD_TOKEN = "•"

class TokenDictionary:
    def __init__(self) -> None:
        self.map = bidict({
            UNKNOWN_TOKEN: 0,
            PAD_TOKEN: 1
        })
        self.next_id = len(self.map)

    def encode(self, token: str, read_only: bool) -> int:
        if read_only:
            id = self.map.get(token, self.map[UNKNOWN_TOKEN])
        else:
            id = self.map.get(token)

            if id is None:
                id = self.next_id
                self.map.put(token, self.next_id)
                self.next_id += 1
            
        return id

    def encode_pad(self) -> int:
        return self.map[PAD_TOKEN]

    def decode(self, id: int) -> str:
        return self.map.inverse[id]