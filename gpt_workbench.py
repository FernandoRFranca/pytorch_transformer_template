import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GPT SECTION


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
    

# TOKENIZER SECTION


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
        self.text = open(data_dir, 'r', encoding='utf-8').read() # load all the data as simple string

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
    

# DATASET SECTION


class GPTDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __getitem__(self, idx):
        return self.tokens[idx]
    
    def __len__(self):
        return self.tokens.shape[0]

    def get_batch(self, start, end):
        return self.tokens[start:end]
    

# DATALOADER SECTION


class FernandoDataLoader:
    def __init__(self, tokens, batch_size, context_length):
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length

        self.current_position = 0

    def get_batch(self):
        # Definir localmente batch_size e context_length
        b, c = self.batch_size, self.context_length

        # Definir posição inicial e final
        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        # Calcular tamanho da janela de tokens iniciais para caso os batchs estejam pegando o final da matriz de tokens
        add_data = -1
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens)
            end_pos = len(self.tokens)

        # Janelar a matriz de tokens
        d = self.tokens[start_pos:end_pos]

        # Se passar do ultimo token, concatenar tokens iniciais
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data]])

        # Capturar x e y, usando reshape para evitar
        x = (d[:-1]).view(b, c)
        y = (d[1:]).view(b, c)
        if add_data != -1:
            self.current_position += b * c
        else:
            self.current_position = 0
        return x, y


class BenchmarkDataLoader:
    def __init__(self, tokens, batch_size, context_length) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length

        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length

        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        # if the batch exceeds total length, get the data till last token
        # and take remaining from starting token to avoid always excluding some data
        add_data = -1 # n, if length exceeds and we need `n` additional tokens from start
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens) - 1
            end_pos = len(self.tokens) - 1

        d = self.tokens[start_pos:end_pos]
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data]])
        x = (d[:-1]).view(b, c)  # inputs
        y = (d[1:]).view(b, c)  # targets

        self.current_position += b * c # set the next position
        return x, y


# TRAINING SECTION


def initialize_model(vocab_size = 100, d_model=512):
    model = FernandoGPT(d_model, vocab_size)
    return model


def train_model(
    m,
    train_loader,
    eval_loader,
    suppress_logits_print=True,
    suppress_intra_epoch_print=True
):
    lr = 1e-3
    optim = torch.optim.AdamW(m.parameters(), lr=lr)

    epochs = 5000
    eval_steps = 1000 # perform evaluation in every n steps
    for ep in range(epochs):
        xb, yb = train_loader.get_batch()

        logits, loss = m(xb, yb)
        if not suppress_logits_print:
            print(f"Training loss: {loss} - Sample of logits: {logits[:1]}")
        elif not suppress_intra_epoch_print:
            print(f"Training loss: {loss}")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if ep % eval_steps == 0 or ep == epochs-1:
            m.eval()
            with torch.no_grad():
                xvb, yvb = eval_loader.get_batch()
                _, e_loss = m(xvb, yvb)

                print(f"Epoch: {ep}\tlr: {lr}\ttrain_loss: {loss}\teval_loss: {e_loss}")
            m.train() # back to training mode
    
    return "Success."


# TEST SECTION


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


def test_model_training():
    train_batch_size = 16  # training batch size
    eval_batch_size = 8  # evaluation batch size
    context_length = 256  # number of tokens processed in a single batch
    train_split = 0.8  # percentage of data to use from total data for training

    data_dir = "dataset/data.txt"
    tokenizer = FernandoTokenizer(data_dir)
    text = tokenizer.text
    print(f"Text sample: {text[:10] if len(text) > 10 else text}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

    # split data into trian and eval
    n_data = len(data)
    train_data = data[:int(n_data * train_split)]
    eval_data = data[int(n_data * train_split):]

    train_loader = FernandoDataLoader(train_data, train_batch_size, context_length)
    eval_loader = FernandoDataLoader(eval_data, eval_batch_size, context_length)

    chars = list(set(text))
    vocab_size = len(chars)
    d_model = vocab_size 

    m = FernandoGPT(vocab_size=vocab_size, d_model=d_model).to(device)
    assert train_model(m, train_loader, eval_loader) == "Success."

    with torch.no_grad():
        input = torch.tensor(tokenizer.encode("Love"), dtype=torch.long, device=device).unsqueeze(0)
        prediction = tokenizer.decode(m.generate(input, max_len=500))
        print(prediction)


def test_tokenizer_instanciation():
    tokenizer = FernandoTokenizer(vocab_dir='dataset/data.txt')
    print("Teste encodificação de token 'H': ", tokenizer.encode('H'))
    print("Teste de decodificação de token idx '1': ", tokenizer.decode(torch.tensor([1]).unsqueeze(0)))
    
    model = initialize_model(vocab_size=100, d_model=100)
    model.to(device=device)
    with torch.no_grad():
        input = torch.tensor(tokenizer.encode("Love"), dtype=torch.long, device=device).unsqueeze(0)
        prediction = tokenizer.decode(model.generate(input, max_len=500))
        print(prediction)


if __name__ == '__main__':
    # test_model_instanciation()
    # test_tokenizer_instanciation()
    test_model_training()