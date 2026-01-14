import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  r2_score
import shap
import matplotlib.pyplot as plt
import pickle


from sklearn.feature_selection import RFE


#set relevant directory locations
global_data_dir ='../data/'
models_dir ='../models/'
figures_dir = '../figures/'


# Load data from CSV
census_in = f'{global_data_dir}inegi_coneval_dataset_2020_common_mesh_20240905.csv'

data = pd.read_csv(census_in)


responses = ["pobreza_2020","pobreza_extrema_2020", "pobreza_moderada_2020", "carencia_social_2020",
 "ingreso_2020", "no_pobres_2020", "rezago_educativo_2020","servicios_salud_2020",
 "seguridad_social_2020", "calidad_vivienda_2020","servicios_basicos_2020","alimentacion_2020",
 "una_carencia_2020","tres_carencias_2020","ingreso_inferior_2020","ingreso_inferior_minimo_2020"]

responses_en = ["poverty", "extreme poverty",  "moderate poverty",  "social deprivation",
                "income", "not poor",  "educational backwardness", "health_services",
                 "social_security",  "housing quality", "basic_services", "nutrition",
                "one deprivation", "three deprivations", "lower income", "minimum lower income"]

responses_lst = ["calidad_vivienda_2020"]


num_iter = 1000

k = 0
X_orig = data.drop(responses + ['cve_mun'], axis=1)
for r in responses:
    print(r)
    y_orig = data[r]

    # Identifying rows with NaN values in either DataFrame
    nan_mask = X_orig.isna().any(axis=1) | y_orig.isna()
    X = X_orig[~nan_mask]
    y = y_orig[~nan_mask]

    # Load the variable back from the file
    #fn_in = data_dir + 'fine_tuned_xgb_model_06.pkl'
    fn_in =f'{models_dir}fine_tuned_xgb_model_{r}_split_1.pkl'
    with open(fn_in, 'rb') as file:
        xgb_model = pickle.load(file)


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#, random_state=42)

    # Normalize data using training set
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the variable to a pickle file
    with open(f'{models_dir}scaler_{r}_split_1.pkl', 'wb') as file:
        pickle.dump(scaler, file)





    # Evaluate the model
    y_pred = xgb_model.predict(X_test, output_margin=True)
    r2 = r2_score(y_test, y_pred)
    print(f"Coefficient of Determination (R^2): {r2}")

    

    explainer = shap.TreeExplainer(xgb_model,X)
    explanation = explainer(X_test)

    shap_values = explanation.values
    # make sure the SHAP values add up to marginal predictions
    max_diff =np.abs(shap_values.sum(axis=1) + explanation.base_values - y_pred).max()
    print(f'Max difference: {max_diff}')


    # Save SHAP values to CSV
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_values_df.to_csv(f'{global_data_dir}shap_values_{r}.csv', index=False)

    # Compute mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values_df).mean(axis=0)

    # Sort the features based on mean absolute SHAP values
    sorted_features = mean_abs_shap_values.sort_values(ascending=False)

    # Save the ordered list of important features to a CSV file
    sorted_features.to_csv(f'{global_data_dir}shap_feature_importance_{r}.csv')




    # Save base values and predictions (optional, for reference)
    base_values_df = pd.DataFrame(explanation.base_values, columns=['base_value'])
    predictions_df = pd.DataFrame(y_pred, columns=['prediction'])
    base_values_df.to_csv(f'{global_data_dir}base_values_{r}.csv', index=False)
    predictions_df.to_csv(f'{global_data_dir}predictions_{r}.csv', index=False)


    # Plot and save SHAP beeswarm plot with black background and white text
    plt.figure(facecolor='black')
    shap.plots.beeswarm(explanation, show=False)
    plt.gca().set_facecolor('black')
    plt.gca().tick_params(colors='white', which='both')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    plt.title(f'SHAP Feature Importance for {r}', color='white')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('white')


    plt.savefig(f'{figures_dir}/shap_beeswarm_plot_{r}.png', bbox_inches='tight', facecolor='black')

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    TEXT_COLOR = "black"  # ← pon "black" si prefieres letras negras
    mpl.rcParams.update({
        "text.usetex": False,  # use LaTeX
        "font.family": "serif",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # Plot and save SHAP summary plot
    fig, ax = plt.subplots(figsize=(6, 8), facecolor='white')
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X.columns,
        show=False,  # build first, then style
        color_bar=False,  # drop the colour bar (it encodes relative feature value)
        plot_size=None,
        plot_type="dot"
    )
    # --- BEESWARM -------------------------------------------------------
    fig = plt.figure(figsize=(6, 8), facecolor="white")
    shap.plots.beeswarm(explanation, show=False, color_bar=False)

    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), style='italic')  # ← NEW LINE

    ax.tick_params(colors=TEXT_COLOR)
    plt.title(f"SHAP Feature Importance for {responses_en[k]}", color=TEXT_COLOR)
    plt.savefig(f"{figures_dir}/shap_beeswarm_plot_{r}.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- SUMMARY --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 8), facecolor="white")
    shap.summary_plot(shap_values, X_test, feature_names=X.columns,
                      show=False, color_bar=False, plot_type="dot")

    plt.setp(ax.get_yticklabels(), style='italic')  # ← NEW LINE

    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title(f"SHAP Summary Plot for {responses_en[k]}", color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/shap_summary_plot_{r}.png",
                dpi=300, facecolor="white")
    plt.close(fig)
    k = k + 1




