import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FernandoGPT(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, inputs, targets=None):
        self.batch_size = inputs.shape[0]
        self.sequence_length = inputs.shape[1]
        logits = self.wte(inputs) # dim -> (batch_size, sequence_length, d_model)
        loss = None
        if targets is not None:
            logits = logits.view(self.batch_size * self.sequence_length, self.d_model)
            y = targets.view(self.batch_size * self.sequence_length)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, inputs, max_len):
        for _ in range(max_len):
            logits, _ = self.forward(inputs)
            logits = logits[:, -1, :] # Get the last token
            logits = nn.Softmax(dim=-1)(logits)
            prediction_token_idx = torch.multinomial(logits, num_samples=1)
            inputs = torch.cat([inputs, prediction_token_idx], dim=1)
        return inputs
        

class BenchmarkGPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
    
    def forward(self, inputs, targets = None):
        logits = self.wte(inputs) # dim -> batch_size, sequence_length, d_model
        loss = None
        if targets != None:
            print(f"Decoder logits shape: {logits.shape}")
            batch_size, sequence_length, d_model = logits.shape
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(inputs)  
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=1)            
            idx_next = torch.multinomial(probs, num_samples=1) 
            inputs = torch.cat([inputs, idx_next], dim=1)
        return inputs
    

class FernandoTokenizer():
    def __init__(self, vocab_dir='text.txt') -> None:
        self.vocab_dir = vocab_dir
        self.text = open(vocab_dir, 'r', encoding='utf-8').read()
        self.chars = sorted(list(set(self.text)))

        self.chr_to_idx = {c: i for i, c in tqdm(enumerate(self.chars), total=len(self.chars), desc='Criando índices do tokenizador...')}
        self.unk_token = '<unk>'
        self.chr_to_idx[self.unk_token] = -1
        print("Dicionário de codificação gerado: ", self.chr_to_idx)
        self.idx_to_chr = {i: c for c, i in tqdm(self.chr_to_idx.items())}
        print("Dicionário de decodificação gerado: ", self.idx_to_chr)

    def encode(self, text):
        idxs = []
        for char in text:
            if char not in self.chars:
                idxs.append(self.chr_to_idx[self.unk_token])
                continue
            idxs.append(self.chr_to_idx[char])
        return idxs
    
    def decode(self, idxs):
        chars = []
        assert idxs.shape[0] == 1, "Erro no tamanho do vetor de saida do modelo. Tamanho não compativel com decodificação do Tokenizer."
        idxs = idxs[0]
        for idx in idxs:
            if idx.item() not in self.idx_to_chr.keys():
                chars.append(self.unk_token)
                continue
            chars.append(self.idx_to_chr[idx.item()])
        return ''.join(chars)
    

class BenchmarkTokenizer():
    def __init__(self, data_dir='data.txt'):
        self.data_dir = "data.txt"
        self.text = open(data_dir, 'r').read() # load all the data as simple string

        # Get all unique characters in the text as vocabulary
        self.chars = list(set(self.text))
        self.vocab_size = len(self.chars)

        # build the character level tokenizer
        self.chr_to_idx = {c:i for i, c in enumerate(self.chars)}
        self.idx_to_chr = {i:c for i, c in enumerate(self.chars)}

    def encode(self, input_text: str) -> list[int]:
        return [self.chr_to_idx[t] for t in input_text]

    def decode(self, input_tokens: list[int]) -> str:
        return "".join([self.idx_to_chr[i] for i in input_tokens])


def initialize_model(vocab_size = 100, d_model=512):
    model = FernandoGPT(d_model, vocab_size)
    return model


def train_model():
    pass


def test_model_instanciation():
    # Test the model
    batch_size = 2
    sequence_length = 10
    vocab_size = 100
    d_model = 100
    model = FernandoGPT(d_model, vocab_size)
    # model = BenchmarkGPT(vocab_size, d_model)
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))
    logits, loss = model(inputs, targets)
    print(logits.shape, loss)
    generated = model.generate(inputs, 30)
    print(generated.shape)
    print("Test complete.")


def test_tokenizer_instanciation():
    tokenizer = FernandoTokenizer(vocab_dir='dataset/data.txt')
    print("Teste encodificação de token 'H': ", tokenizer.encode('H'))
    print("Teste de decodificação de token idx '1': ", tokenizer.decode(torch.tensor([1]).unsqueeze(0)))
    
    model = initialize_model(vocab_size=100, d_model=100)
    model.to(device=device)
    with torch.no_grad():
        input = torch.tensor(tokenizer.encode("Love"), dtype=torch.long, device=device).unsqueeze(0)
        prediction = tokenizer.decode(model.generate(input, max_len=10000))
        print(prediction)


if __name__ == '__main__':
    # test_model_instanciation()
    test_tokenizer_instanciation()