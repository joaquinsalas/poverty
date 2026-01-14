import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os


# Set relevant directory locations
common_data_dir = '../INEGI_CPV2020_n9/'
global_data_dir = '../data/'
models_dir = '../models/'

# Define the Neural Network model (same as used in the individual NN model)
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

# Define the Ensemble Neural Network model
class EnsembleNet(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units):
        super(EnsembleNet, self).__init__()
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

def load_scalers(r, repeat, models_dir):
    # Load scalers for each model
    scaler_nn_path = os.path.join(models_dir, f'scaler_nn_{r}_split_{repeat+1}.pkl')
    scaler_xgb_path = os.path.join(models_dir, f'scaler_xgb_{r}_split_{repeat+1}.pkl')
    scaler_svr_path = os.path.join(models_dir, f'scaler_svr_{r}_split_{repeat+1}.pkl')

    with open(scaler_nn_path, 'rb') as file:
        scaler_nn = pickle.load(file)
    with open(scaler_xgb_path, 'rb') as file:
        scaler_xgb = pickle.load(file)
    with open(scaler_svr_path, 'rb') as file:
        scaler_svr = pickle.load(file)
    return scaler_nn, scaler_xgb, scaler_svr

def load_models(r, repeat, input_size, models_dir):
    # Load each model (NN, XGB, SVR)
    nn_model_path = os.path.join(models_dir, f'best_nn_model_{r}_split_{repeat+1}.pth')
    nn_model_param_path = os.path.join(models_dir, f'best_nn_params_{r}_split_{repeat+1}.pkl')
    xgb_model_path = os.path.join(models_dir, f'fine_tuned_xgb_model_{r}_split_{repeat+1}.pkl')
    svr_model_path = os.path.join(models_dir, f'fine_tuned_svr_model_{r}_split_{repeat+1}.pkl')

    with open(nn_model_param_path, 'rb') as file:
        nn_model_param = pickle.load(file)

    nn_model = Net(input_size=input_size, hidden_layers=nn_model_param['num_layers'],
                   hidden_units=nn_model_param['num_neurons'])
    nn_model.load_state_dict(torch.load(nn_model_path))
    nn_model.eval()

    with open(xgb_model_path, 'rb') as file:
        xgb_model = pickle.load(file)

    with open(svr_model_path, 'rb') as file:
        svr_model = pickle.load(file)

    return nn_model, xgb_model, svr_model

def train_ensemble_model(model, dataloader_train, criterion, optimizer, num_epochs, patience, X_val_tensor, y_val_tensor):
    best_rmse = float('inf')
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

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
            best_model_state = model.state_dict()  # Save best model state
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break

    return best_model_state, best_val_loss

def prepareDataset(X, y, scaler_nn, scaler_xgb, scaler_svr, nn_model, xgb_model, svr_model, repeat):
    # Split data into training (50%), validation (20%), and test sets (30%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=repeat)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2857, random_state=repeat)

    # Normalize data using each scaler
    X_nn_scaled_train = scaler_nn.transform(X_train)
    X_nn_scaled_test = scaler_nn.transform(X_test)
    X_nn_scaled_val = scaler_nn.transform(X_val)

    X_xgb_scaled_train = scaler_xgb.transform(X_train)
    X_xgb_scaled_test = scaler_xgb.transform(X_test)
    X_xgb_scaled_val = scaler_xgb.transform(X_val)

    X_svr_scaled_train = scaler_svr.transform(X_train)
    X_svr_scaled_test = scaler_svr.transform(X_test)
    X_svr_scaled_val = scaler_svr.transform(X_val)

    # Get predictions from individual models
    nn_preds_train = nn_model(torch.tensor(X_nn_scaled_train, dtype=torch.float32)).detach().numpy()
    nn_preds_test = nn_model(torch.tensor(X_nn_scaled_test, dtype=torch.float32)).detach().numpy()
    nn_preds_val = nn_model(torch.tensor(X_nn_scaled_val, dtype=torch.float32)).detach().numpy()

    xgb_preds_train = xgb_model.predict(X_xgb_scaled_train)
    xgb_preds_test = xgb_model.predict(X_xgb_scaled_test)
    xgb_preds_val = xgb_model.predict(X_xgb_scaled_val)

    svr_preds_train = svr_model.predict(X_svr_scaled_train)
    svr_preds_test = svr_model.predict(X_svr_scaled_test)
    svr_preds_val = svr_model.predict(X_svr_scaled_val)

    # Combine predictions as inputs for the ensemble model
    X_ensemble_train = np.column_stack((nn_preds_train, xgb_preds_train, svr_preds_train))
    X_ensemble_test = np.column_stack((nn_preds_test, xgb_preds_test, svr_preds_test))
    X_ensemble_val = np.column_stack((nn_preds_val, xgb_preds_val, svr_preds_val))

    X_train_tensor = torch.tensor(X_ensemble_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_ensemble_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_ensemble_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    return (dataloader_train, X_train_tensor, y_train_tensor,
        X_val_tensor,   y_val_tensor,
        X_test_tensor,  y_test_tensor,
        nn_preds_test,  xgb_preds_test, svr_preds_test)

# Set hyperparameters for the grid search
layers = [1, 2, 3, 4]
neurons = range(3, 21)  # Neurons per layer from 3 to 20
num_epochs = 1000
patience = 5

# Load data from CSV
census_in = os.path.join(global_data_dir, 'inegi_coneval_dataset_2020_common_mesh_20240905.csv')
data = pd.read_csv(census_in)

responses = ["pobreza_2020", "pobreza_extrema_2020", "pobreza_moderada_2020", "carencia_social_2020",
             "ingreso_2020", "no_pobres_2020", "rezago_educativo_2020", "servicios_salud_2020",
             "seguridad_social_2020", "calidad_vivienda_2020", "servicios_basicos_2020", "alimentacion_2020",
             "una_carencia_2020", "tres_carencias_2020", "ingreso_inferior_2020", "ingreso_inferior_minimo_2020"]

num_repeats = 30

best_model_params = {}

# Initialize a list to store results for all responses
all_results = []

# Loop through each response
for r in responses:
    print(f"Response {r}")

    # Prepare data
    X_orig = data.drop(responses + ['cve_mun'], axis=1)
    y_orig = data[r]

    nan_mask = X_orig.isna().any(axis=1) | y_orig.isna()
    X = X_orig[~nan_mask]
    y = y_orig[~nan_mask]

    results = []  # Store results for each response

    for repeat in range(num_repeats):  # Repeat the process 30 times for each response

        scaler_nn, scaler_xgb, scaler_svr = load_scalers(r, repeat, models_dir)
        nn_model, xgb_model, svr_model = load_models(r, repeat, X.shape[1], models_dir)

        (dataloader_train, X_train, y_train,
         X_val, y_val,
         X_test, y_test,
         nn_preds_test, xgb_preds_test, svr_preds_test) = prepareDataset(X, y,
                                                                        scaler_nn, scaler_xgb, scaler_svr,
                                                                        nn_model, xgb_model, svr_model,
                                                                        repeat)

        best_model = None
        best_val_loss = float('inf')
        best_model_state = None
        r2_scores = []

        for num_layers in layers:
            for num_neurons in neurons:
                model = EnsembleNet(X_train.shape[1], num_layers, num_neurons)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                model_state, val_loss = train_ensemble_model(model, dataloader_train, criterion, optimizer, num_epochs,
                                                             patience, X_val, y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model_state
                    best_model_params[r] = {'num_layers': num_layers, 'num_neurons': num_neurons}
                    # Save the best model state to a file
                    model_save_path = os.path.join(models_dir, f'best_ensemble_model_{r}_split_{repeat + 1}.pth')
                    torch.save(best_model_state, model_save_path)
                    # Save the best parameters to a file
                    params_save_path = os.path.join(models_dir, f'best_ensemble_params_{r}_split_{repeat + 1}.pkl')
                    with open(params_save_path, 'wb') as file:
                        pickle.dump(best_model_params[r], file)


        # Evaluate on test data with the best model
        model = EnsembleNet(X_train.shape[1], best_model_params[r]['num_layers'],
                            best_model_params[r]['num_neurons'])
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test).numpy()
            r2_ens = r2_score(y_test, y_pred_test)

        y_true = y_test.numpy().ravel()  # ground-truth
        r2_nn = r2_score(y_true, nn_preds_test.ravel())
        r2_xgb = r2_score(y_true, xgb_preds_test.ravel())
        r2_svr = r2_score(y_true, svr_preds_test.ravel())

        # Append the results for this repeat
        results.append({
            "response": r,
            "repeat": repeat,
            "R2_NN": r2_nn,
            "R2_XGB": r2_xgb,
            "R2_SVR": r2_svr,
            "R2_Ensemble": r2_ens
        })

        print(f"R2 for {r} on repeat {repeat + 1}: Ens {r2_ens:.4f}, NN {r2_nn:.4f}, XGB {r2_xgb:.4f}, SVR {r2_svr:.4f}")

    # Store results for all repetitions of this response
    all_results.extend(results)

# Convert all results to a DataFrame and save as CSV
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv(os.path.join(global_data_dir, 'ensemble_nn_results_30_repeats_per_response_20250627.csv'), index=False)

print(f"Results saved to {os.path.join(global_data_dir, 'ensemble_nn_results_30_repeats_per_response_20250627.csv')}")

