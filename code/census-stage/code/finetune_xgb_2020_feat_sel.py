import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import warnings

# Set relevant directory locations
global_data_dir = '../data/'
models_dir = '../models/'

# Load data from CSV
census_in = global_data_dir + 'inegi_coneval_dataset_2020_common_mesh_20240905.csv'
data = pd.read_csv(census_in)

# Load the feature importance file and filter features based on a threshold
importance_file = global_data_dir + 'shap_feature_importance_calidad_vivienda_2020.csv'
importance_df = pd.read_csv(importance_file)


# Get unique importance values as thresholds
thresholds = importance_df['importance'].unique()
thresholds.sort()
thresholds = thresholds[::-1]  # Reverse the sorted array for descending order


responses = ["calidad_vivienda_2020"]

# Define hyperparameter search space
param_dist = {
    'eta': np.linspace(0.01, 0.3, 100),
    'colsample_bytree': np.linspace(0.1, 1, 100),
    'max_depth': np.arange(1, 7),
    'subsample': np.linspace(0.1, 1, 100),
    'gamma': np.linspace(0, 1, 100),
}


# Suppress all warnings
warnings.simplefilter(action='ignore', category=Warning)

num_iter = 10000

# Initialize a list to store results
results = []

# Loop through each response and perform 30 different splits
for r in responses:

    tau = 0
    for threshold in thresholds:

        selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()

        # Retain only the selected features in the dataset
        X_orig = data[selected_features]
        y_orig = data[r]

        # Identifying rows with NaN values in either DataFrame
        nan_mask = X_orig.isna().any(axis=1) | y_orig.isna()
        X = X_orig[~nan_mask]
        y = y_orig[~nan_mask]

        for split in range(20):
            print(f"response {r}, threshold {threshold:0.4f} on split {split + 1}")
            # Split data into 50% training and 50% testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

            # Normalize data using the training set
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Random search with XGBRegressor
            model = XGBRegressor(n_estimators=200, objective='reg:squarederror')
            random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=num_iter,
                                           scoring='neg_mean_squared_error', cv=3, verbose=0, n_jobs=-1)
            random_search.fit(X_train, y_train)

            # Best model based on RandomizedSearchCV
            best_model = XGBRegressor(**random_search.best_params_, n_estimators=400, objective='reg:squarederror')
            best_model.fit(X_train, y_train)

            # Evaluate R² on the test set
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            print(f"R2 for {r}, threshold {threshold:0.2f} on split {split + 1}: {r2:.4f}")

            # Save the results for this response and split
            results.append({
                "response": r,
                "threshold": threshold,
                "split": split + 1,
                "R2": r2,
                "eta": random_search.best_params_['eta'],
                "colsample_bytree": random_search.best_params_['colsample_bytree'],
                "max_depth": random_search.best_params_['max_depth'],
                "subsample": random_search.best_params_['subsample'],
                "gamma": random_search.best_params_['gamma']})

            # Save the best model
            model_out = models_dir + f'fine_tuned_xgb_feat_sel_model_{r}_thr_{tau:02d}_split_{split + 1}.pkl'
            with open(model_out, 'wb') as file:
                pickle.dump(best_model, file)

            # Save the scaler
            scaler_out = models_dir + f'scaler_xgb_feat_sel_{r}_thr_{tau:02d}_split_{split + 1}.pkl'
            with open(scaler_out, 'wb') as file:
                pickle.dump(scaler, file)
        tau += 1
# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(global_data_dir + 'xgb_feat_sel_results_30_splits.csv', index=False)

print(f"Results saved to {global_data_dir + 'xgb_feat_sel_results_30_splits.csv'}")

