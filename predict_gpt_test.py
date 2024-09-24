import torch
from model import GPT, BPETokenizer  # Certifique-se de que os caminhos estão corretos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Carregar o modelo treinado
vocab_size = 300  # Substitua pelo tamanho do vocabulário usado no treinamento
d_model = 512  # Substitua pelo valor usado no treinamento
nhead = 8  # Substitua pelo valor usado no treinamento
num_layers = 6  # Substitua pelo valor usado no treinamento

model = GPT(vocab_size, d_model, nhead, num_layers)
checkpoint = torch.load('gpt_weights/checkpoint_19.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inicializar o tokenizer
tokenizer = BPETokenizer(vocab_size=vocab_size)
tokenizer.train(['dataset/ptb_train.txt'])  # Substitua pelo caminho correto do dataset

# Exemplo de uso
input_text = 'first national bank of boston for example is offering' # "Bom dia! Eu gostaria de um café com leite, por favor. Claro, vai querer algo para acompanhar?"
predicted_text = model.predict(input_text, tokenizer)
print("Predicted Text:", predicted_text)
