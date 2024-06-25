from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np



class Mental_CNNGRU(nn.Module):
    def __init__(self, input_dim=30, cnn_out_channels=256, cnn_kernel_size=4, gru_hidden_size=64, output_size=1, dropout_prob=0.5):
        super(Mental_CNNGRU,self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim,out_channels=cnn_out_channels, kernel_size=cnn_kernel_size)
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=gru_hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(gru_hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self,x) :
        x = x.permute(0,2,1)
        x = F.relu(self.conv1d(x))
        x = x.permute(0,2,1)
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = x[:,-1,:]
        x = self.fc(x)
        x = self.activation(x)
        return x

# # Define your custom Model    
# class yourModelName(nn.Module):


# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, val_loader, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        best_val_loss = float('inf')
        best_model = model.state_dict()

        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
            
            # Validation loop
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}')

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

            model.load_state_dict(best_model)
        return model

    return custom_train_torch

# torch test
def test_torch():
    def custom_test_torch(model, test_loader, cfg=None):
        criterion = nn.MSELoss()
        model.eval()
        test_losses = []
        all_targets = []
        all_predictions = []

        print("Starting evaluation...")
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                test_losses.append(loss.item())
                all_targets.extend(targets.numpy())
                all_predictions.extend(outputs.numpy())

        average_loss = sum(test_losses) / len(test_losses)
        accuracy = 0.0  # Accuracy metric can be defined based on the specific task, if required
        # f1 = f1_score(all_targets, [1 if pred > 0.5 else 0 for pred in all_predictions])  # Example threshold for binary classification

        metrics = {"loss": average_loss}

        print(f'Mean Test Loss: {average_loss:.4f}')
        print(f'loss: {average_loss:.4f}')

        model.to("cpu")  # move model back to CPU
        return average_loss, accuracy, metrics

    return custom_test_torch