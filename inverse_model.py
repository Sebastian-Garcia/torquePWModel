import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from ding_model import ding_subject_parameters
import matplotlib.pyplot as plt
import copy

# Select device, ensure it's CUDA 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# load data
data = np.load("fes_dataset_33hz/fes_dataset.npz", allow_pickle=True)
torques = data["torque"]
params = data["params"]
stable_torques = torques.max(axis=1)
pw = params[:, 0] # gets pulse widths


# get ding subject parameters, index 3 is a_scale, 4 is pd0
a_scales = dict([(subject, ding_subject_parameters(subject)[3]) for subject in range(1,11)])
pd0s = dict([(subject, ding_subject_parameters(subject)[4]) for subject in range(1,11)])
a_scales[0] = ding_subject_parameters("average")[3]
pd0s[0] = ding_subject_parameters("average")[4]



class TorqueFESNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)




# prepare data , stable_torques = N length, pw = N length
HELD_OUT = 8
mask = (stable_torques > 0.01) & (params[:,2] != 0) & (params[:,2] != HELD_OUT)
mask_s10 = (stable_torques > 0.01) & (params[:,2] == HELD_OUT)
subj_s10 = params[mask_s10, 2].astype(int)
#X_s10 = np.column_stack([stable_torques[mask_s10],[pd0s[s] for s in subj_s10],[a_scales[s] for s in subj_s10]])  
X_s10 = np.column_stack([stable_torques[mask_s10],[pd0s[s] for s in subj_s10]])  
y_s10 = pw[mask_s10].reshape(-1, 1)



secondParam = np.array([pd0s[int(subject)] for subject in params[:, 2]])
thirdParam = np.array([a_scales[int(subject)] for subject in params[:, 2]])
filtered_torque = stable_torques[mask]
filtered_pw = pw[mask]
filtered_secondParam = secondParam[mask]
#filtered_thirdParam = thirdParam[mask]

# MULTI-INPUT TORQUE AND A_SCALE/PD0
#X = np.column_stack([filtered_torque, filtered_secondParam, filtered_thirdParam])
X = np.column_stack([filtered_torque, filtered_secondParam])
y = filtered_pw.reshape(-1, 1)

# Split data into training, testing, and validation sets (0.7, 0.15, 0.15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42) # 70% training
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=0.5, shuffle=True,random_state=42) # 15% testing, 15% validation




# Create scalers to scale data
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaler.fit(X_train)
y_scaler.fit(y_train)

# apply scale to all data, and convert to tensor for training
X_train_scaled = torch.tensor(x_scaler.transform(X_train), dtype=torch.float32).to(device)
X_test_scaled = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32).to(device)
X_val_scaled = torch.tensor(x_scaler.transform(X_val) , dtype=torch.float32).to(device)
y_train_scaled = torch.tensor(y_scaler.transform(y_train), dtype=torch.float32).to(device)
y_test_scaled = torch.tensor(y_scaler.transform(y_test), dtype=torch.float32).to(device)
y_val_scaled = torch.tensor(y_scaler.transform(y_val), dtype=torch.float32).to(device)


# Convert into tensor dataset objects, 
train_dataset = TensorDataset(X_train_scaled, y_train_scaled)
test_dataset = TensorDataset(X_test_scaled, y_test_scaled)
val_dataset = TensorDataset(X_val_scaled, y_val_scaled)

# Put into dataloaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


# Simple model for now... TODO: change to class-based model definition for later
model = TorqueFESNetwork().to(device)
print("Model Initialized")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)


train_losses = []
val_losses = []
# Training loop
num_epochs = 15000

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150)
best_val_loss = float('inf')
best_weights = None

for epoch in range(num_epochs):
    
    # training pass
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad() #zero the parameter gradients

        # forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        running_loss+= loss.item()
    
    # validation pass
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item()


    
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)   # ← here
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_weights = copy.deepcopy(model.state_dict())



    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch + 1} training_loss={running_loss/len(train_loader):.6f}")
        print(f"val_loss={val_loss/len(val_loader):.6f}")


plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (scaled)")
plt.yscale("log")
plt.title("Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.expanduser("~/fes_project/loss_curve.png"), dpi=120)
plt.show()


# TEST LOOP!!!

model.load_state_dict(best_weights)
print(f"Restored best model (val_loss={best_val_loss:.6f})")


model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        test_loss += criterion(y_pred, y_batch).item()
    

test_mse = test_loss / len(test_loader)
test_rmse_us = np.sqrt(test_mse) * y_scaler.scale_[0]
print(f"Test loss (scaled MSE): {test_mse:.4f}")
print(f"Test RMSE (real units): {test_rmse_us:.1f} µs")



# ── Fine-tuning learning curve on held-out subject ───────────────
torch.save(best_weights, "base_model.pt")

X_s10_scaled = x_scaler.transform(X_s10)
y_s10_scaled = y_scaler.transform(y_s10)

n_values = [1, 3, 5, 10, 20, 50, 100]
rmse_results = {}

for n in n_values:
    idx  = np.random.choice(len(X_s10), n, replace=False)
    rest = np.setdiff1d(np.arange(len(X_s10)), idx)

    ft_model = TorqueFESNetwork().to(device)
    ft_model.load_state_dict(torch.load("base_model.pt"))
    opt_ft = optim.Adam(ft_model.parameters(), lr=1e-4)

    Xft = torch.tensor(X_s10_scaled[idx], dtype=torch.float32).to(device)
    yft = torch.tensor(y_s10_scaled[idx], dtype=torch.float32).to(device)

    for _ in range(500):
        ft_model.train()
        opt_ft.zero_grad()
        criterion(ft_model(Xft), yft).backward()
        opt_ft.step()

    ft_model.eval()
    with torch.no_grad():
        Xtest = torch.tensor(X_s10_scaled[rest], dtype=torch.float32).to(device)
        preds = y_scaler.inverse_transform(ft_model(Xtest).cpu().numpy())
    rmse = np.sqrt(np.mean((preds - y_s10[rest]) ** 2))
    rmse_results[n] = rmse
    print(f"N={n:4d}  RMSE={rmse:.2f} µs")

plt.figure()
plt.plot(list(rmse_results.keys()), list(rmse_results.values()), marker='o')
plt.xlabel("Fine-tuning samples (N)")
plt.ylabel("RMSE (µs)")
plt.title(f"Subject {HELD_OUT} — fine-tuning learning curve")
plt.grid(True)
plt.savefig(os.path.expanduser("~/fes_project/finetune_curve.png"), dpi=120)
plt.show()
