import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal, uniform
import pickle
import warnings
import platform

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set relevant directory locations based on the operating system
if platform.system() == "Windows":
    code_dir = 'E:/Documents/informs/research/2023.11.16census/code/'
    local_data_dir = 'E:/Documents/informs/research/2023.11.16census/data/'
    common_data_dir = 'E:/Documents/informs/research/2023.11.16census/2024.07.29malla470/INEGI_CPV2020_n9/'
    global_data_dir = 'E:/Documents/informs/research/2023.11.16census/2024.07.29malla470/data/'
    models_dir = 'E:/Documents/informs/research/2023.11.16census/2024.07.29malla470/models/'
else:
    # Exxact
    code_dir = '/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/code/'
    local_data_dir = '/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/data/'
    common_data_dir = '/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/INEGI_CPV2020_n9/'
    global_data_dir = '/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/data/'
    models_dir = '/mnt/data-r1/JoaquinSalas/Documents/informs/research/2023.11.16census/2024.07.29malla470/models/'

# Load data from CSV
census_in = global_data_dir + 'inegi_coneval_dataset_2020_common_mesh_20240905.csv'
data = pd.read_csv(census_in)

responses = ["pobreza_2020", "pobreza_extrema_2020", "pobreza_moderada_2020", "carencia_social_2020",
             "ingreso_2020", "no_pobres_2020", "rezago_educativo_2020", "servicios_salud_2020",
             "seguridad_social_2020", "calidad_vivienda_2020", "servicios_basicos_2020", "alimentacion_2020",
             "una_carencia_2020", "tres_carencias_2020", "ingreso_inferior_2020", "ingreso_inferior_minimo_2020"]

# Define hyperparameter search space
param_distributions = {
    "C": reciprocal(0.01, 10),  # More flexibility in regularization strength
    "gamma": expon(scale=0.01),  # Focus on smaller gamma values, useful for RBF and poly kernels
    "kernel": ["rbf", "linear"], # Degree for polynomial kernel
    "degree": [2, 3, 4, 5],  # Degree for polynomial kernel
    "coef0": uniform(-1, 1),  # Coefficient for poly/sigmoid kernel
    "epsilon": uniform(0.001, 0.1),  # Epsilon-tube for regression
    "shrinking": [True, False],  # Whether to use shrinking heuristic
    "tol": uniform(1e-3, 1e-2),  # Tolerance for stopping criterion
    "max_iter": [5000, 10000, 20000],  # Increase max_iter
}

# Initialize a list to store results
results = []

# Suppress all warnings
warnings.simplefilter(action='ignore', category=Warning)

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
        # Split data into 50% training and 50% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        # Normalize data using the training set
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize the SVR model
        svr = SVR()

        # Random search with SVR
        random_search = RandomizedSearchCV(
            estimator=svr,
            param_distributions=param_distributions,
            n_iter=100,
            cv=3,
            scoring="r2",
            verbose=0,
            n_jobs=-1
        )

        # Suppress warnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            random_search.fit(X_train, y_train)

        # Train the best model
        best_model = random_search.best_estimator_

        # Evaluate R² on the test set
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"R2 for {r} on split {split + 1}: {r2:.4f}")

        # Save the results for this response and split
        results.append({
            "response": r,
            "split": split + 1,
            "R2": r2,
            "C": random_search.best_params_['C'],
            "coef0": random_search.best_params_['coef0'],
            "degree": random_search.best_params_['degree'],
            "epsilon": random_search.best_params_['epsilon'],
            "gamma": random_search.best_params_['gamma'],
            "kernel": random_search.best_params_['kernel'],
            "max_iter": random_search.best_params_['max_iter'],
            "shrinking": random_search.best_params_['shrinking'],
            "tol": random_search.best_params_['tol']
        })

        # Save the best model
        fn_out = models_dir + f'fine_tuned_svr_model_{r}_split_{split + 1}.pkl'
        with open(fn_out, 'wb') as file:
            pickle.dump(best_model, file)

        # Save the best parameters
        fn_out = models_dir + f'best_param_svr_model_{r}_split_{split + 1}.pkl'
        with open(fn_out, 'wb') as file:
            pickle.dump(random_search.best_params_, file)

        # Save the scaler
        scaler_out = models_dir + f'scaler_svr_{r}_split_{split + 1}.pkl'
        with open(scaler_out, 'wb') as file:
            pickle.dump(scaler, file)

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(global_data_dir + 'svr_results_30_splits.csv', index=False)

print(f"Results saved to {models_dir + 'svr_results_30_splits.csv'}")
