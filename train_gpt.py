import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from model import GPT, TokenizerGPT, TextDataset, DataLoaderGPT

# Set the random seed for reproducibility
torch.manual_seed(42)


def train_gpt(
        n_epochs=5,
        vocab_size=50000,
        d_model=512,
        nhead=8,
        num_layers=6,
        batch_size=8,
        lr=10-4,
        sequence_max_len=4096,
        use_subsampled_dataset=False,
        n_samples=1000
    ):
    # Set the device
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
        torch.cuda.empty_cache()
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
    tokenizer = TokenizerGPT(vocab_size=vocab_size)
    tokenizer.train([data_path])

    # Initialize the model
    print("Initializing the model...")

    # Checks if a checkpoint exists
    checkpoint_path = "gpt_weights/checkpoint_{last_epoch}.pt"
    for i in range(n_epochs, 0, -1):
        if os.path.exists(checkpoint_path.format(last_epoch=i)):
            checkpoint_path = checkpoint_path.format(last_epoch=i)
            break

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model = GPT(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = Adam(model.parameters(), lr=lr, eps=1e-9)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model = GPT(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        for param in model.parameters():
            param.requires_grad = True

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

    # Initialize the dataset
    print("Initializing the dataset...")
    dataset = TextDataset(data_path, tokenizer, max_len=sequence_max_len)

    if use_subsampled_dataset:
        # Define the sample size
        sample_size = n_samples  # Adjust this to the desired sample size

        # Slice the dataset to the sample size
        dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:sample_size])

    # Split the dataset into training and validation sets
    print("Splitting the dataset into training and validation sets...")
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoaderGPT(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoaderGPT(val_dataset, batch_size=1, shuffle=False)

    # Initialize the optimizer
    print("Initializing the optimizer...")
    optimizer = Adam(model.parameters(), lr=10**-4, eps=1e-9)

    # Initialize the loss function
    print("Initializing the loss function...")
    loss_fn = nn.CrossEntropyLoss()

    # Define the training loop
    print("Starting training...")
    for epoch in range(n_epochs):  # Number of epochs
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # Move data to the same device as the model
            batch = batch.to(device)

            # Create src and tgt sequences
            src = batch[:, :-1]  # All but the last token
            tgt = batch[:, 1:]  # All but the first token

            # Forward pass
            outputs = model(src)

            # Compute loss
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), tgt.reshape(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # Save a checkpoint
        print("Saving a checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"gpt_weights/checkpoint_{epoch+1}.pt")

        # Validation step
        max_batch_idx = 5
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = batch.to(device)
                src = batch[:, :-1]  # All but the last token
                tgt = batch[:, 1:]  # All but the first token
                outputs = model(src)
                loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), tgt.reshape(-1))
                print(f"Validation Loss: {loss.item()}")

                # Print out some predictions
                predictions = torch.argmax(outputs, dim=-1)
                predicted_tokens = predictions.tolist()
                for tokens in predicted_tokens:
                    predicted_words = tokenizer.decode(tokens)
                    print("Predictions:", predicted_words)
                if batch_idx == max_batch_idx:
                    break

    print("Training complete.")
    return model


if __name__ == "__main__":
    train_gpt(
        n_epochs=3,
        vocab_size=50000,
        d_model=512,
        nhead=8,
        num_layers=6,
        batch_size=8,
        lr=10-4,
        sequence_max_len=256,
        use_subsampled_dataset=True,
        n_samples=100000
    )