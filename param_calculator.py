vocab_size = 50000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
output_vocab_size = 50000
sequence_length = 100  # Assuming a sequence length of 100 for positional encoding

# Embeddings
embeddings = (vocab_size + sequence_length) * d_model

# Encoder and Decoder layers
encoder_parameters = num_layers * (4 * d_model**2 + 2 * d_model * d_ff)
decoder_parameters = num_layers * (6 * d_model**2 + 2 * d_model * d_ff)

# Output Layer
output_layer = d_model * output_vocab_size

total_parameters = embeddings + encoder_parameters + decoder_parameters + output_layer
print(f"Total number of parameters: {total_parameters}")
