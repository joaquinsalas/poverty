import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import platform

# Set relevant directory locations based on the operating system
if platform.system() == "Windows":
    code_dir ='E:/Documents/informs/research/2023.11.16census/code/'
    local_data_dir ='E:/Documents/informs/research/2023.11.16census/data/'
    common_data_dir ='E:/Documents/informs/research/2023.11.16census/2024.07.29malla470/INEGI_CPV2020_n9/'
    global_data_dir ='E:/Documents/informs/research/2023.11.16census/2024.07.29malla470/data/'
    models_dir ='E:/Documents/informs/research/2023.11.16census/2024.07.29malla470/models/'
else:
    # Exxact
    code_dir ='/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/code/'
    local_data_dir ='/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/data/'
    common_data_dir ='/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/INEGI_CPV2020_n9/'
    global_data_dir ='/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/data/'
    models_dir ='/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/models/'


# Load data from CSV
census_in = global_data_dir + 'inegi_coneval_dataset_2020_common_mesh_20240905.csv'
data = pd.read_csv(census_in)

responses = ["pobreza_2020", "pobreza_extrema_2020", "pobreza_moderada_2020", "carencia_social_2020",
             "ingreso_2020", "no_pobres_2020", "rezago_educativo_2020", "servicios_salud_2020",
             "seguridad_social_2020", "calidad_vivienda_2020", "servicios_basicos_2020", "alimentacion_2020",
             "una_carencia_2020", "tres_carencias_2020", "ingreso_inferior_2020", "ingreso_inferior_minimo_2020"]

# Set hyperparameters for the grid search
layers = [1, 2, 3, 4]
neurons = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
num_epochs = 1000
patience = 5

# Define the Neural Network model
class Net(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units):
        super(Net, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize a list to store results
results = []

X_orig = data.drop(responses + ['cve_mun'], axis=1)

# Loop through each response and perform 30 different splits
for r in responses:
    y_orig = data[r]

    # Identifying rows with NaN values in either DataFrame
    nan_mask = X_orig.isna().any(axis=1) | y_orig.isna()
    X = X_orig[~nan_mask]
    y = y_orig[~nan_mask]

    for split in range(30):
        print(f"response {r} on split {split + 1}")
        # Split data into training (50%), validation (20%), and test sets (30%)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2857)  # 0.2857 ≈ 20% of original data

        # Normalize data using the training set
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

        best_rmse = float('inf')
        best_model = None
        best_model_params = {}
        early_stopping_counter = 0
        best_val_loss = float('inf')

        for num_layers in layers:
            for num_neurons in neurons:
                model = Net(X_train.shape[1], num_layers, num_neurons)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(num_epochs):
                    model.train()
                    for inputs, targets in dataloader_train:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                    # Evaluate on validation data
                    model.eval()
                    with torch.no_grad():
                        y_pred_val = model(X_val_tensor)
                        val_loss = criterion(y_pred_val, y_val_tensor).item()

                    # Early stopping logic
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= patience:
                        break

                if best_val_loss < best_rmse:
                    best_rmse = best_val_loss
                    best_model = model
                    best_model_params = {'num_layers': num_layers, 'num_neurons': num_neurons}

        # Save the best model
        model_path = os.path.join(models_dir, f'best_nn_model_{r}_split_{split + 1}.pth')
        torch.save(best_model.state_dict(), model_path)

        # Save the best parameters
        params_path = os.path.join(models_dir, f'best_nn_params_{r}_split_{split + 1}.pkl')
        with open(params_path, 'wb') as file:
            pickle.dump(best_model_params, file)

        # Save the scaler
        scaler_path = os.path.join(models_dir, f'scaler_nn_{r}_split_{split + 1}.pkl')
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)

        # Calculate R² score on test data
        best_model.eval()
        with torch.no_grad():
            y_pred_test = best_model(X_test_tensor).numpy()
            r2 = r2_score(y_test, y_pred_test)
            results.append({
                "response": r,
                "split": split + 1,
                "R2": r2,
                "num_layers": best_model_params['num_layers'],
                "num_neurons": best_model_params['num_neurons']
            })
            print(f"Best R2 for {r} on split {split + 1}: {r2:.4f}")

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(global_data_dir, 'nn_results_30_splits.csv'), index=False)

print(f"Results saved to {os.path.join(global_data_dir, 'nn_results_30_splits.csv')}")




