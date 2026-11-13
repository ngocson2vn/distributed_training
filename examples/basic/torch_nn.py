import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader  # 1. Import the new tools


class PyTorchNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=2, out_features=2)
        self.output = nn.Linear(in_features=2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


def train(epochs = 10001):
    dataset = torch.tensor(
        [
            [15, 3, 1],
            [10, 5, 1],
            [20, 1, 1],
            [1, 5, 0],
            [5, 0, 0],
            [30, 0, 1],
            [2, 1, 0],
            [5, 5, 1],
            [7, 10, 0],
            [25, 6, 1],
        ],
        dtype=torch.float32,
    )

    X = dataset[:, :2]
    y_true = dataset[:, 2].view(-1, 1)

    # 2. Wrap your X and y tensors in a Dataset
    torch_dataset = TensorDataset(X, y_true)

    # 3. Create a DataLoader to handle batching and shuffling
    # batch_size=4 means the model will process 4 rows at a time
    dataloader = DataLoader(torch_dataset, batch_size=4, shuffle=True)

    model = PyTorchNeuralNetwork()
    # When batching, we usually just let PyTorch calculate the mean loss for the batch
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 4. Update the training loop with a nested batch loop
    for epoch in range(epochs):
        epoch_loss = 0.0  # Track total loss to calculate the epoch average

        # The DataLoader automatically yields batches of X and y!
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            loss.backward()
            optimizer.step()

            # Accumulate the loss (multiply by batch size to get the un-averaged sum)
            epoch_loss += loss.item() * batch_X.size(0)

        # Calculate the true average loss across the entire dataset for this epoch
        avg_loss = epoch_loss / len(torch_dataset)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")

        if avg_loss < 1e-3:
            break

if __name__ == "__main__":
    train(epochs=1000001)
