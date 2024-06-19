import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataset import random_split

from model import GPT, TokenizerGPT, TextDataset, DataLoaderGPT

# Set the random seed for reproducibility
torch.manual_seed(42)


def train_model_gpt():
    # Set the device
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

    # Define the model, tokenizer, dataset, optimizer and loss function
    data_path = "dataset/enwik9.txt"

    # Initialize the tokenizer and train it
    print("Training the tokenizer...")
    tokenizer = TokenizerGPT(vocab_size=50000)
    tokenizer.train([data_path])

    # Initialize the model
    print("Initializing the model...")
    model = GPT(vocab_size=50000, d_model=512, nhead=8, num_layers=6)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

    # Initialize the dataset
    print("Initializing the dataset...")
    dataset = TextDataset(data_path, tokenizer, max_len=4096)

    # Split the dataset into training and validation sets
    print("Splitting the dataset into training and validation sets...")
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoaderGPT(train_dataset, batch_size=32)
    val_loader = DataLoaderGPT(val_dataset, batch_size=32)

    # Initialize the optimizer
    print("Initializing the optimizer...")
    optimizer = Adam(model.parameters())

    # Initialize the loss function
    print("Initializing the loss function...")
    loss_fn = nn.CrossEntropyLoss()

    # Define the training loop
    print("Starting training...")
    for epoch in range(10):  # Number of epochs
        for batch in train_loader:
            # Move data to the same device as the model
            batch = batch.to(device)

            # Create src and tgt sequences
            src = batch[:, :-1]  # All but the last token
            tgt = batch[:, 1:]  # All but the first token

            # Forward pass
            outputs = model(src, tgt)

            # Compute loss
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), tgt.view(-1))

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

    print("Training complete.")
    return model


if __name__ == "__main__":
    train_model_gpt()