import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pickle
import platform
import os

# Set relevant directory locations
global_data_dir = '../data/'
models_dir = '../models/'

# Load feature importance file and filter features based on a threshold
importance_file = global_data_dir + 'shap_feature_importance_calidad_vivienda_2020.csv'
importance_df = pd.read_csv(importance_file)

# Set importance threshold (example: 0.01, adjust as needed)
threshold = 0.002550
selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
num_features = len(selected_features)

# Load the dataset
census_in = global_data_dir + 'inegi_coneval_dataset_2020_common_mesh_20240905.csv'
data = pd.read_csv(census_in)

# Keep only selected features more comments
X = data[selected_features]

 

# Define the response variable
responses = ["calidad_vivienda_2020"]

final_results = pd.DataFrame({'codigo': data['codigo']})  # Initialize with 'codigo'
# Iterate over each response variable
for r in responses:

    for split in range(30):
        print(f"response {r} on split {split + 1}")

        scaler_xgb_path = os.path.join(models_dir, f'scaler_xgb_feat_sel_{r}_thr_{num_features-1:02d}_split_{split + 1}.pkl')
        with open(scaler_xgb_path, 'rb') as file:
            scaler_xgb = pickle.load(file)

        xgb_model_path = os.path.join(models_dir, f'fine_tuned_xgb_model_{r}_split_thr_{num_features-1:02d}_{split + 1}.pkl')
        with open(xgb_model_path, 'rb') as file:
            xgb_model = pickle.load(file)

        X_xgb_scaled_test = scaler_xgb.transform(X)

        xgb_preds = xgb_model.predict(X_xgb_scaled_test)



        # Save predictions for this repeat
        final_results[f'prediction_{r}_{split:02d}'] = xgb_preds



# Save the final results to a CSV file
final_results.to_csv(f'../data/xgb_feat_sel_inferences_20241015.csv', index=False)


