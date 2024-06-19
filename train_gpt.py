import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataset import random_split

from model import GPT, TokenizerGPT, TextDataset, DataLoaderGPT

# Set the random seed for reproducibility
torch.manual_seed(42)

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model, tokenizer, dataset, optimizer and loss function
data_path = "/path/to/your/data.txt"

# Initialize the tokenizer and train it
tokenizer = TokenizerGPT(vocab_size=50000)
tokenizer.train([data_path])

# Initialize the model
model = GPT(vocab_size=50000, d_model=512, nhead=8, num_layers=6)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

# Initialize the dataset
dataset = TextDataset(tokenizer, data_path)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoaderGPT(train_dataset, batch_size=32)
val_loader = DataLoaderGPT(val_dataset, batch_size=32)

# Initialize the optimizer
optimizer = Adam(model.parameters())

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the training loop
print("Starting training...")
for epoch in range(10):  # Number of epochs
    for batch in train_loader:  # Replace with your own data loader
        # Move data to the same device as the model
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Validation step
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            print(f"Validation Loss: {loss.item()}")

            # Print out some predictions
            predictions = torch.argmax(outputs, dim=-1)
            print("Predictions:", predictions)

    # Save a checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"gpt_weights/checkpoint_{epoch+1}.pt")