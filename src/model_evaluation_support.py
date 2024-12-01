# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import math

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Para realizar clasificacion y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve
    
)
from sklearn.model_selection import cross_validate

from sklearn.feature_selection import RFECV

import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


import xgboost as xgb
import pickle

import shap


# Gestionar los warnings
# -----------------------------------------------------------------------
import warnings

# modificar el path
# -----------------------------------------------------------------------
import sys
import os
sys.path.append("..")

# crear archivos temporales
# -----------------------------------------------------------------------
import tempfile

# statistics functions
# -----------------------------------------------------------------------
from scipy.stats import  norm


# Registro de modelos
# -----------------------------------------------------------------------
import mlflow
import mlflow.sklearn


# Acceder a parámetros de métodos de objetos
# -----------------------------------------------------------------------
import inspect


seed = 42


def model_evaluation_CV_run(run_name, models, scores, X, y, crossval, verbose=False):
    results = []

    n_splits = crossval.get_n_splits(X=X, y=y)

    warnings.filterwarnings('ignore')

    for name, model in models:
        # Start a child run for each model

        if verbose:
            print(f"\nTraining {name}.")
        # Cross_val
        cv_results = cross_validate(model, X, y, cv=crossval, scoring=scores, verbose=verbose, return_train_score=True)

        # Store results for each fold and each metric
        for split in range(n_splits):
            result = {"Model": name, "Split": split + 1}
            for score in scores:
                # prepare results_df
                result[f"test_{score}"] = cv_results[f"test_{score}"][split]
                result[f"train_{score}"] = cv_results[f"train_{score}"][split]
                
            results.append(result)
        

        if verbose:
            # Print results numerically
            print(f'--\n{name} model:')
            for score in scores:
                print("%s: mean %f, std (%f) " % (score, cv_results[f"test_{score}"].mean(), cv_results[f"test_{score}"].std()))


        results_df = pd.DataFrame(results)

    warnings.filterwarnings('default')


    return results_df


def run_gridsearch_experiment(X_train, y_train, model_name, model, param_grid, cross_val, score, verbosity):
    # Dynamically set verbosity if supported
    # if 'verbose' in inspect.signature(model.__init__).parameters:
    #     model.verbose = verbosity  


    grid_search = GridSearchCV(model, 
                            param_grid, 
                            cv=cross_val, 
                            scoring=score, 
                            n_jobs = -1,
                            verbose=verbosity)

    grid_search.fit(X_train, y_train)
    return {
        "model_name": model_name,
        "pipeline": grid_search.best_estimator_,
        "params": grid_search.best_params_,
        "score": grid_search.best_score_
    }


def run_bayessearch_experiment(X_train, y_train, model_name, model, param_space, cross_val, n_iter, score,verbosity, seed =42):
    pass
    # # Dynamically set verbosity if supported
    # if 'verbose' in inspect.signature(model.__init__).parameters:
    #     model.verbose = verbosity  
    
    # bayes_search = BayesSearchCV(estimator=model,
    #                             search_spaces=param_space, 
    #                             n_iter=n_iter, 
    #                             cv=cross_val, 
    #                             scoring=score,
    #                             n_jobs=-1,
    #                             random_state=seed)

    # bayes_search.fit(X_train, y_train)
    # return {
    #     "model_name": model_name,
    #     "pipeline": bayes_search.best_estimator_,
    #     "params": bayes_search.best_params_,
    #     "score": bayes_search.best_score_
    # }



def plot_prediction_residuals(y_test, y_test_pred):

    fig, axes = plt.subplots(2,2, figsize=(15,8))
    axes = axes.flat

    sns.histplot(y_test_pred, bins=30, ax=axes[0], label="y_test_pred")
    sns.histplot(y_test, bins=30, ax=axes[0], alpha=0.25, label="y_test")
    axes[0].legend()
    axes[0].set_title('y_test vs y_test_pred distribution')

    axes[1].scatter(y_test.reset_index(drop=True), y_test_pred, color='purple')

    axes[1].axline((0, 0), slope=1, color='r', linestyle='--')
    axes[1].set_title('y_test Vs. y_pred')
    axes[1].set_xlabel('True values')
    axes[1].set_ylabel('Predicted values')


    residuals = y_test.values - y_test_pred

    sns.histplot(residuals, bins=30, ax=axes[2])
    axes[2].set_title('Residuals distribution')

    # Plot residuals
    axes[3].scatter(y_test_pred, residuals, color='purple')
    axes[3].axhline(0, linestyle='--', color='black', linewidth=1)
    axes[3].set_title('Residuals Plot')
    axes[3].set_xlabel('Predicted Values')
    axes[3].set_ylabel('Residuals')



    plt.tight_layout()
    plt.show()


def test_evaluate_model(run_name, model, X_train, y_train, X_test, y_test, best_params=None, tag=None,train_multiplier=1,test_multiplier=1):
    modelo = model.set_params(**(best_params or {}))

    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train) * train_multiplier
    y_test_pred = modelo.predict(X_test) * test_multiplier
    y_train = y_train*train_multiplier
    y_test = y_test*test_multiplier

    metricas = calculate_train_test_metrics(y_train, y_test, y_train_pred, y_test_pred)

    resultados_df = pd.DataFrame(metricas).T

    plot_prediction_residuals(y_test, y_test_pred)

    return resultados_df



def log_dataframe_to_mlflow(dataframe: pd.DataFrame, artifact_path: str):
    """
    Save a DataFrame as a temporary CSV file and log it as an artifact in MLflow.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be saved and logged.
    artifact_path : str
        The artifact path within MLflow where the file will be stored.
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp_results.csv")
        
        # Save DataFrame to the temporary file
        dataframe.to_csv(temp_file_path, index=False)
        
        # Log with MLflow
        mlflow.log_artifact(temp_file_path, artifact_path=artifact_path)
        



def calcular_metricas(self, modelo_nombre):
    """
    Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC, Kappa,
    tiempo de computación y núcleos utilizados.
    """
    if modelo_nombre not in self.resultados:
        raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
    
    pred_train = self.resultados[modelo_nombre]["pred_train"]
    pred_test = self.resultados[modelo_nombre]["pred_test"]

    if pred_train is None or pred_test is None:
        raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
    
    modelo = self.resultados[modelo_nombre]["mejor_modelo"]

    # Registrar tiempo de ejecución
    start_time = time.time()
    if hasattr(modelo, "predict_proba"):
        prob_train = modelo.predict_proba(self.X_train)[:, 1]
        prob_test = modelo.predict_proba(self.X_test)[:, 1]
    else:
        prob_train = prob_test = None
    elapsed_time = time.time() - start_time

    # Registrar núcleos utilizados
    num_nucleos = psutil.cpu_count(logical=True)

    # Métricas para conjunto de entrenamiento
    metricas_train = {
        "accuracy": accuracy_score(self.y_train, pred_train),
        "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
        "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
        "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
        "kappa": cohen_kappa_score(self.y_train, pred_train),
        "auc": roc_auc_score(self.y_train, prob_train) if prob_train is not None else None,
        "time_seconds": elapsed_time,
        "n_jobs": num_nucleos
    }

    # Métricas para conjunto de prueba
    metricas_test = {
        "accuracy": accuracy_score(self.y_test, pred_test),
        "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
        "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
        "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
        "kappa": cohen_kappa_score(self.y_test, pred_test),
        "auc": roc_auc_score(self.y_test, prob_test) if prob_test is not None else None,
        "time_seconds": elapsed_time,
        "n_jobs": num_nucleos
    }

    # Combinar métricas en un DataFrame
    return pd.DataFrame({"train": metricas_train, "test": metricas_test}).T

def plot_matriz_confusion(modelo_nombre, y_test, y_pred):
    """
    Plotea la matriz de confusión para el modelo seleccionado.
    """

    # Matriz de confusión
    matriz_conf = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
    plt.title(f"Matriz de Confusión ({modelo_nombre})")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.show()
    
def importancia_predictores(self, modelo_nombre):
    """
    Calcula y grafica la importancia de las características para el modelo seleccionado.
    """
    if modelo_nombre not in self.resultados:
        raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
    
    modelo = self.resultados[modelo_nombre]["mejor_modelo"]
    if modelo is None:
        raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
    
    # Verificar si el modelo tiene importancia de características
    if hasattr(modelo, "feature_importances_"):
        importancia = modelo.feature_importances_
    elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
        importancia = modelo.coef_[0]
    else:
        print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
        return
    
    # Crear DataFrame y graficar
    importancia_df = pd.DataFrame({
        "Feature": self.X.columns,
        "Importance": importancia
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
    plt.title(f"Importancia de Características ({modelo_nombre})")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.show()

def plot_shap_summary(self, modelo_nombre):
    """
    Genera un SHAP summary plot para el modelo seleccionado.
    Maneja correctamente modelos de clasificación con múltiples clases.
    """
    if modelo_nombre not in self.resultados:
        raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

    modelo = self.resultados[modelo_nombre]["mejor_modelo"]

    if modelo is None:
        raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de generar el SHAP plot.")

    # Usar TreeExplainer para modelos basados en árboles
    if modelo_nombre in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(self.X_test)

        # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
        if isinstance(shap_values, list):
            # Para modelos binarios, seleccionar SHAP values de la clase positiva
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            # Para Decision Trees, seleccionar SHAP values de la clase positiva
            shap_values = shap_values[:, :, 1]
    else:
        # Usar el explicador genérico para otros modelos
        explainer = shap.Explainer(modelo, self.X_test, check_additivity=False)
        shap_values = explainer(self.X_test).values

    # Generar el summary plot estándar
    shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns)

# Función para asignar colores
def color_filas_por_modelo(row):
    if row["modelo"] == "tree":
        return ["background-color: #e6b3e0; color: black"] * len(row)  
    
    elif row["modelo"] == "random_forest":
        return ["background-color: #c2f0c2; color: black"] * len(row) 

    elif row["modelo"] == "gradient_boosting":
        return ["background-color: #ffd9b3; color: black"] * len(row)  

    elif row["modelo"] == "xgboost":
        return ["background-color: #f7b3c2; color: black"] * len(row)  

    elif row["modelo"] == "logistic_regression":
        return ["background-color: #b3d1ff; color: black"] * len(row)  
    
    return ["color: black"] * len(row)


def calcular_ic(df: pd.DataFrame, columna: str, alpha: float, metodo: str ="normal", seed: int = None, n_bootstrap: int = 1000) -> tuple:
    """
    Calcula el intervalo de confianza al 95% para una columna de un DataFrame
    utilizando la distribución normal.
    
    Args:
        data (pd.DataFrame): Dataset con los datos.
        columna (str): Nombre de la columna para calcular el IC.
        
    Returns:
        tuple: (límite_inferior, límite_superior) del intervalo de confianza.
    """
    # verificar que la columna exista
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el dataframe.")
       
    if metodo == "normal":
        # Calcular estadísticos
        media = df[columna].mean()
        desviacion_estandar = df[columna].std()
        n = len(df[columna])
        
        # definir % IC
        ci = 1 - (alpha) / 2
        z = norm.ppf(ci) 
        
        # limites del intervalo
        margen_error = z * (desviacion_estandar / np.sqrt(n))
        limite_inferior = round(float(media - margen_error),2)
        limite_superior = round(float(media + margen_error),2)

    
    elif metodo == "bootstrap":
        if seed is not None:
            np.random.seed(seed)
        
        # Generar muestras bootstrap
        bootstrap_means = []
        n = len(df[columna])
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(df[columna], size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        
        # Calcular percentiles para el IC
        limite_inferior = round(float(np.percentile(bootstrap_means, alpha / 2 * 100)), 2)
        limite_superior = round(float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100)), 2)
    
    else:
        raise ValueError("El método debe ser 'normal' o 'bootstrap'.")
    
    print(f"El RMSE del test se encuentra con un {100 * (1 - alpha):.1f}% de confianza entre {limite_inferior} y {limite_superior}")
    
    return limite_inferior, limite_superior

def select_best_features(X_train, y_train, score, cross_val, model="linear_regression", params=None, method="grid"):
    if model == "decision_tree":
        if method == "grid":
            search = GridSearchCV(
                estimator=DecisionTreeRegressor(),
                param_grid= (params or {}),
                scoring=score,
                cv=cross_val,
                n_jobs=-1
            )
        elif method == "bayes": 
            search = BayesSearchCV(
                estimator=DecisionTreeRegressor(),
                search_spaces= (params or {}),
                scoring=score,
                cv=cross_val,
                n_jobs=-1,
                random_state=seed
            )
        else: 
            raise ValueError("The method introduced is not valid. It must be either 'grid' or 'bayes'.")
        
        search.fit(X_train, y_train)

        # Get the best tree
        model = search.best_estimator_

    elif model == "linear_regression":
        model = LinearRegression()
        
    else:
        raise ValueError("The model introduced is not valid. It must be either 'linear_regression' or 'decision_tree'")

    rfecv = RFECV(estimator=model, cv=cross_val, scoring=score, n_jobs=-1)

    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_]

    print(f"Selected Features with RFECV: {selected_features}")

    return selected_features

def run_pipelines(X_train, y_train, preprocessing_pipeline, models, cross_val, score, verbosity, search_method="grid"):
    best_pipelines = []
    for model_name, (model, params) in models.items():
        pipeline = Pipeline(preprocessing_pipeline.steps + [('classifier', model)])
        
        if search_method == "grid":
            result = run_gridsearch_experiment(X_train=X_train, y_train=y_train, model_name=model_name, model=pipeline, 
                                               param_grid=params, cross_val=cross_val, score=score, verbosity=verbosity)
        elif search_method == "bayes":
            result = run_bayessearch_experiment(X_train=X_train, y_train=y_train, model_name=model_name, n_iter=30,
                                                model=pipeline, param_space=params, cross_val=cross_val, score=score, verbosity=verbosity)
        else: 
            raise ValueError("The search_method introduced is not valid")
        
        best_pipelines.append(result)

    best_model = max(best_pipelines, key=lambda x: x["score"])

    if verbosity:
        print("Best Model Overall:")
        print(f"Model Name: {best_model['model_name']}")
        print(f"Best Score: {best_model['score']}")
        print(f"Best Parameters: {best_model['params']}")

    return best_model, best_pipelines


def plot_score_by_threshold_multiple(model_names_list, y_test, y_probs_list, scorer_list=cohen_kappa_score, scorer_name_list='Cohen\'s Kappa'):
    thresholds = [i / 100 for i in range(1, 100)]

    scorer_dict = {}
    for scorer_name, scorer in zip(scorer_name_list, scorer_list):
        scorer_dict[scorer_name] = {model_name: {"scorer": scorer, "scores": []} for model_name in model_names_list}

    n_cols = 2
    n_rows = math.ceil(len(scorer_list)/2)
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15, 5*n_rows))
    axes = axes.flat

    fig.suptitle("Threshold metric comparison")

    for ax, scorer_name in zip(axes, scorer_name_list):
        for model_name, y_prob in zip(model_names_list, y_probs_list):
            for threshold in thresholds:
                preds = (y_prob >= threshold).astype(int)
                scorer_dict[scorer_name][model_name]["scores"].append(
                    scorer_dict[scorer_name][model_name]["scorer"](y_test, preds)
                )

            sns.lineplot(x=thresholds, y=scorer_dict[scorer_name][model_name]["scores"], label=model_name, ax=ax)

        ax.set_xlabel('Threshold')
        ax.set_ylabel(f'{scorer_name}')
        ax.set_title(f'{scorer_name} vs Threshold')
    
    if len(scorer_list) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.show()


def plot_auc_and_aucpr(model_name, y_true, y_prob):
    # calcular AUC 
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    # calcular AUC-PR
    auc_pr = average_precision_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    # plotear ambas curvas en una figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # curva ROC
    ax1.plot(fpr, tpr, label=f'{model_name} curva ROC (AUC = {auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Curva ROC para {model_name}')
    ax1.legend(loc='best')

    # curva AUC-PR
    ax2.plot(recall, precision, label=f'{model_name} curva Precision-Recall (AUC-PR = {auc_pr:.4f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Curva Precision-Recall para {model_name}')
    ax2.legend(loc='best')
    

    plt.tight_layout()
    plt.show()
    
    # imprimir valores test de AUC y AUC-PR
    print(f'{model_name} - test AUC: {auc:.4f}')
    print(f'{model_name} - test AUC-PR: {auc_pr:.4f}')


def find_optimal_threshold(y_true, y_probs, scoring_function, thresholds=None):
    """
    Finds the threshold that maximizes the specified scoring function.

    Parameters:
    - y_true: np.array or list, true binary labels (0s and 1s).
    - y_probs: np.array or list, predicted probabilities.
    - scoring_function: callable, function to compute the score (e.g., precision_score).
    - thresholds: np.array, optional, thresholds to evaluate (default is np.linspace(0, 1, 100)).

    Returns:
    - optimal_threshold: float, the threshold that maximizes the score.
    - max_score: float, the maximum score achieved.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)  # Default thresholds from 0 to 1
    
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        scores.append(scoring_function(y_true, y_pred))
    
    max_score_index = np.argmax(scores)
    max_score = scores[max_score_index]
    optimal_threshold = thresholds[max_score_index]
    
    return optimal_threshold, max_score


def maximize_revenue(y_true, y_probs, thresholds=None):
    """
    Finds the threshold that maximizes the custom objective function.

    Parameters:
    - y_true: np.array or list, true binary labels (0s and 1s).
    - y_probs: np.array or list, predicted probabilities.
    - thresholds: np.array, optional, thresholds to evaluate (default is np.linspace(0, 1, 100)).

    Returns:
    - optimal_threshold: float, the threshold that maximizes the objective function.
    - max_objective: float, the maximum value of the objective function.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100) 
    
    # start minimum revenue from loss
    max_revenue = float('-inf')
    optimal_threshold = None
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        # calculate the objective function maximize saving
        revenue = (60 * TP) - (70 * FN) - (10 * FP)
        
        # update max_revenue
        if revenue > max_revenue:
            max_revenue = revenue
            optimal_threshold = thresh
    
    return optimal_threshold, max_revenue
