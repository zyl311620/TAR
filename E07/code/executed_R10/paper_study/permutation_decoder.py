"""Token mask for exact JSON permutations; no response repair or label information.

The grammar accepts each unused ordinal once, including multi-digit IDs. It
evaluates complete decoded token pieces, so punctuation/number token boundaries
cannot bypass the uniqueness check. Only ASCII pieces are admitted. Whitespace
between JSON symbols is optional and bounded, preventing endless whitespace.
"""
from dataclasses import dataclass
from functools import lru_cache

PROTOCOL = 'native-exact-permutation-json-v1-20261106'
_PREFIX = '{"ranking":['
_ALPHABET = frozenset(_PREFIX + '0123456789,]} \n\r\t')

@dataclass(frozen=True)
class State:
    phase: str
    remaining: tuple
    offset: int = 0
    digits: str = ''
    whitespace: bool = False

class PermutationGrammar:
    def __init__(self, count):
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError('The registered ranking head has 1..100 candidates')
        self.initial = State('prefix', tuple(str(i) for i in range(1,count+1)))

    @lru_cache(maxsize=32768)
    def character(self, state, char):
        if state.phase == 'done': return None
        if char in ' \n\r\t':
            allowed = (state.phase in ('number','separator','close') or
                       state.phase == 'prefix' and state.offset in (0,1,10,11))
            if not allowed or state.whitespace: return None
            if state.phase == 'number' and state.digits:
                if state.digits not in state.remaining: return None
                return State('separator',state.remaining,digits=state.digits,whitespace=True)
            return State(state.phase,state.remaining,state.offset,state.digits,True)
        if state.phase == 'prefix':
            if char != _PREFIX[state.offset]: return None
            offset = state.offset+1
            return State('number',state.remaining) if offset == len(_PREFIX) else State('prefix',state.remaining,offset)
        if state.phase in ('number','separator'):
            if state.phase == 'number' and char in '0123456789':
                digits = state.digits+char
                if not any(n.startswith(digits) for n in state.remaining): return None
                return State('number',state.remaining,digits=digits)
            if state.digits not in state.remaining: return None
            remaining = tuple(n for n in state.remaining if n != state.digits)
            if char == ',' and remaining: return State('number',remaining)
            if char == ']' and not remaining: return State('close',())
            return None
        if state.phase == 'close' and char == '}': return State('done',())
        return None

    @lru_cache(maxsize=32768)
    def piece(self, state, text):
        for char in text:
            state = self.character(state,char)
            if state is None: return None
        return state

class PermutationTokenVocabulary:
    def __init__(self, tokenizer, eos_token_ids):
        self.eos = tuple(dict.fromkeys(eos_token_ids))
        if not self.eos: raise ValueError('Missing configured EOS')
        special = set(tokenizer.all_special_ids)
        self.pieces = {}
        self.trie = {}
        for token in range(len(tokenizer)):
            if token in special: continue
            text = tokenizer.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if not text or not set(text) <= _ALPHABET: continue
            self.pieces[token] = text
            node = self.trie
            for char in text: node = node.setdefault(char,{})
            node.setdefault(None,[]).append(token)

    def constraint(self, count, prompt_length):
        return PermutationConstraint(self,count,prompt_length)

class PermutationConstraint:
    def __init__(self, vocabulary, count, prompt_length):
        self.vocabulary = vocabulary
        self.grammar = PermutationGrammar(count)
        self.prompt_length = prompt_length
        self.states = {():self.grammar.initial}
        self.allowed_cache = {}

    def state(self, tokens):
        if tokens in self.states: return self.states[tokens]
        # Incremental processing avoids decoding/tokenizing the full prefix at
        # every generation step. No model outputs are edited after generation.
        parent = self.state(tokens[:-1])
        piece = self.vocabulary.pieces.get(tokens[-1])
        next_state = self.grammar.piece(parent,piece) if piece is not None else None
        if next_state is None: raise ValueError('Generated prefix violated the permutation grammar')
        self.states[tokens] = next_state
        return next_state

    def allowed(self, state):
        if state.phase == 'done': return list(self.vocabulary.eos)
        if state in self.allowed_cache: return self.allowed_cache[state]
        output=[]
        def walk(node,current):
            output.extend(node.get(None,()))
            for char,child in node.items():
                if char is None: continue
                after = self.grammar.character(current,char)
                if after is not None: walk(child,after)
        walk(self.vocabulary.trie,state)
        if not output: raise ValueError('No legal next token; fail closed without unmasking')
        self.allowed_cache[state]=output
        return output

    def __call__(self, batch_id, input_ids):
        if batch_id != 0: raise ValueError('The authorized worker uses batch size one')
        tokens=tuple(input_ids[self.prompt_length:].tolist())
        return self.allowed(self.state(tokens))
