import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------ Load splits ------------
X_train = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_train.npy").astype(np.float32)
y_train = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_train.npy").astype(np.int64)
X_val   = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_val.npy").astype(np.float32)
y_val   = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_val.npy").astype(np.int64)
X_test  = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_test.npy").astype(np.float32)
y_test  = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_test.npy").astype(np.int64)

num_classes = len(np.unique(y_train))

# ------------ Dataset ------------
class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)   # (N,T,F)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SignDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader  = DataLoader(SignDataset(X_test, y_test), batch_size=32, shuffle=False)

# ------------ Model ------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=126, hidden=128, num_layers=2, num_classes=6, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)      # (B,T,2H)
        feat = out[:, -1, :]       # last timestep
        return self.fc(feat)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTMClassifier(input_dim=126, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ------------ Train loop ------------
def eval_loader(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * len(yb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return loss_sum/total, correct/total

best_val = 0
patience, wait = 8, 0

for epoch in range(1, 61):  # up to 60 epochs
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = eval_loader(train_loader)
    val_loss, val_acc = eval_loader(val_loader)

    print(f"Epoch {epoch:02d} | train acc {train_acc:.3f} val acc {val_acc:.3f}")

    # early stopping
    if val_acc > best_val:
        best_val = val_acc
        wait = 0
        torch.save(model.state_dict(), "/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ------------ Test ------------
model.load_state_dict(torch.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm.pth", map_location=device))
test_loss, test_acc = eval_loader(test_loader)
print("\nTEST ACC:", test_acc)
