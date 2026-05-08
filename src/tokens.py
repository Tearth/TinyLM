import logging
import operator

from bidict import ON_DUP_DEFAULT, bidict
import regex

UNKNOWN_TOKEN = "⊗"
WHITESPACE_TOKEN = " "
NEWLINE_TOKEN = "\n"
TAG_BEGIN = "<"
TAG_END = ">"

class Word:
    def __init__(self, content: str, count: int) -> None:
        if content.startswith(TAG_BEGIN) and content.endswith(TAG_END):
            self.tokens = [content]
            self.tag = True
        else:
            self.tokens = list(content)
            self.tag = False
        self.count = count

class TokenDictionary:
    def __init__(self) -> None:
        self.map = bidict({
            UNKNOWN_TOKEN: 0,
            WHITESPACE_TOKEN: 1,
            NEWLINE_TOKEN: 2
        })
        self.next_id = len(self.map)

    def build(self, path: str, vocabulary_size: int):
        words = {}
        pairs = {}
        vocabulary = set(self.map)

        # Split dataset and prepare a dictionary of all possible words, symbols and tags
        with open(path, encoding="utf-8") as file:
            for line in file:
                for substring in regex.findall(rf'{TAG_BEGIN}\/?[^{TAG_END}]+\/?{TAG_END}|[\p{{L}}]+|\d+|[\p{{P}}\p{{S}}]', line):
                    if substring not in words:
                        words[substring] = Word(substring, 1)
                    else:
                        words[substring].count += 1

        # Calculate the initial dictionary with pairs
        for word in words.values():
            if word.tag:
                continue

            for i in range(0, len(word.tokens) - 1):
                pair = (word.tokens[i], word.tokens[i + 1])
                
                if pair not in pairs:
                    pairs[pair] = word.count
                else:
                    pairs[pair] += word.count

        # Merge iteratively the most frequent pairs until the desired token dictionary size is achieved
        while len(vocabulary) < vocabulary_size:
            best_pair = max(pairs.items(), key=operator.itemgetter(1))
            best_pair_key = best_pair[0]
            best_pair_value = best_pair[1]
            best_pair_key_merged = best_pair_key[0] + best_pair_key[1]

            logging.debug(f"Vocabulary size: {len(vocabulary)}/{vocabulary_size}, Best pair: {best_pair_key}, Value: {best_pair_value}")

            if best_pair_value == 1:
                break

            vocabulary = set(self.map)

            for word in words.values():
                for i in reversed(range(0, len(word.tokens) - 1)):
                    if (word.tokens[i], word.tokens[i + 1]) == best_pair_key:
                        # Merge the previous token with a new pair
                        if i > 0:
                            previous_pair = (word.tokens[i - 1], word.tokens[i])
                            new_pair = (word.tokens[i - 1], best_pair_key_merged)
                            pairs[previous_pair] -= word.count

                            if pairs[previous_pair] == 0:
                                pairs.pop(previous_pair)

                            if new_pair not in pairs:
                                pairs[new_pair] = word.count
                            else:
                                pairs[new_pair] += word.count
                        
                        # Merge the next token with a new pair
                        if i < len(word.tokens) - 2:
                            next_pair = (word.tokens[i + 1], word.tokens[i + 2])
                            new_pair = (best_pair_key_merged, word.tokens[i + 2])
                            pairs[next_pair] -= word.count

                            if pairs[next_pair] == 0:
                                pairs.pop(next_pair)

                            if new_pair not in pairs:
                                pairs[new_pair] = word.count
                            else:
                                pairs[new_pair] += word.count

                        word.tokens = word.tokens[:i] + [best_pair_key_merged] + word.tokens[i + 2:]
                        pairs[best_pair_key] -= word.count

                        if pairs[best_pair_key] == 0:
                            pairs.pop(best_pair_key)
                
                for token in word.tokens:
                    vocabulary.add(token)

        # Fill token dictionary 
        for token in vocabulary:
            if self.map.get(token) is None:
                self.map.put(token, self.next_id)
                self.next_id += 1

    def encode_token(self, token: str) -> list[int]:
        token_ids = []
        start = 0
        end = len(token)

        if token == WHITESPACE_TOKEN:
            return [self.encode_whitespace()]
        
        if token == NEWLINE_TOKEN:
            return [self.encode_newline()]

        while start < end and end != 0:
            id = self.map.get(token[start:end])
            if id is None:
                end -= 1
            else:
                token_ids.append(id)
                start = end
                end = len(token)
            
            if end == 0:
                token_ids.append(self.encode_unknown_token())

        return token_ids

    def encode_block(self, block: str) -> list[int]:
        token_ids = []

        for line in block.splitlines(True):
            for word in regex.findall(rf'{TAG_BEGIN}\/?[^{TAG_END}]+\/?{TAG_END}|[\p{{L}}]+|\d+|[\p{{P}}\p{{S}}]|\s', line):
                token_ids.extend(self.encode_token(word))

        return token_ids 

    def encode_unknown_token(self) -> int:
        return self.map[UNKNOWN_TOKEN]
    
    def encode_whitespace(self) -> int:
        return self.map[WHITESPACE_TOKEN]
    
    def encode_newline(self) -> int:
        return self.map[NEWLINE_TOKEN]

    def decode_token(self, id: int) -> str:
        return self.map.inverse[id]