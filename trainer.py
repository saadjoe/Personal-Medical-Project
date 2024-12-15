import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, dataloader, device, epochs, learning_rate):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm for a progress bar
        batch = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in batch:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm loop with current loss
            batch.set_postfix(loss=loss.item())

    return model
