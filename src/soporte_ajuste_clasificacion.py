# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

import time
import psutil

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    average_precision_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import catboost as catb
import pickle

import shap

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer

import os


class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente, iteracion="default", seed=42):
        self.dataframe = dataframe.copy()
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=seed, shuffle=True, stratify=self.y,
        )
        self.seed = seed
        self.iteracion = iteracion
        self.categorical = self.X_train.select_dtypes(include=["object"]).columns.to_list()

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(random_state=self.seed),
            "decision_tree": DecisionTreeClassifier(random_state=self.seed),
            "random_forest": RandomForestClassifier(n_jobs=-1,random_state=self.seed),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.seed),
            "xgboost": xgb.XGBClassifier(n_jobs=-1),
            "catboost": catb.CatBoostClassifier(thread_count=-1, random_state=seed, 
                                                cat_features=self.categorical, early_stopping_rounds=50,
                                                eval_metric="PRAUC", custom_metric="PRAUC", verbose=100)
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, 
                                    "pred_test": None, "pred_test_prob": None, 
                                    "best_score":None, "best_params": None,
                                    "mean_fit_time": None,"mean_score_time": None} for nombre in self.modelos}

    def ajustar_modelo(self, modelo_nombre, preprocessing_pipeline, param_grid=None, cross_validation = 5, score="average_precision", pipeline=False):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo
        # parametros_default = {
        #     "logistic_regression": [
        #         {'penalty': ['l1'], 'solver': ['saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
        #         {'penalty': ['l2'], 'solver': ['liblinear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
        #         {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
        #         {'penalty': [None], 'solver': ['lbfgs'], 'max_iter': [10000]}
        #     ],
        #     "tree": {
        #         'max_depth': [3, 5, 7, 10],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4]
        #     },
        #     "random_forest": {
        #         'n_estimators': [50, 100, 200],
        #         'max_depth': [None, 10, 20, 30],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4],
        #         'max_features': ['auto', 'sqrt', 'log2']
        #     },
        #     "gradient_boosting": {
        #         'n_estimators': [100, 200],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 4, 5],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4],
        #         'subsample': [0.8, 1.0]
        #     },
        #     "xgboost": {
        #         'n_estimators': [100, 200],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 4, 5],
        #         'min_child_weight': [1, 3, 5],
        #         'subsample': [0.8, 1.0],
        #         'colsample_bytree': [0.8, 1.0]
        #     }
        # }

        parametros_default = {
            "logistic_regression": [
                {'penalty': ['l1'], 'solver': ['saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['l2'], 'solver': ['liblinear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': [None], 'solver': ['lbfgs'], 'max_iter': [10000]}
            ],
                "decision_tree": [
                    {
                        'max_depth': [3, 5, 7, 10],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    },
                    {
                        'max_depth': [5, 10],
                        'min_samples_split': [10],
                        'min_samples_leaf': [4]
                    }
                ],
                "random_forest": [
                    {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                    },
                    {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, 30],
                        'min_samples_split': [5, 10],
                        'min_samples_leaf': [2, 4],
                        'max_features': ['sqrt', 'log2']
                    }
                ],
                "gradient_boosting": [
                    {
                        'n_estimators': [100],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 4],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                        'subsample': [1.0]
                    },
                    {
                        'n_estimators': [200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [4, 5],
                        'min_samples_split': [5, 10],
                        'min_samples_leaf': [2, 4],
                        'subsample': [0.8]
                    }
                ],
                "xgboost": [
                    {
                        'n_estimators': [100],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 4],
                        'min_child_weight': [1, 3],
                        'subsample': [1.0],
                        'colsample_bytree': [1.0]
                    },
                    {
                        'n_estimators': [200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [4, 5],
                        'min_child_weight': [3, 5],
                        'subsample': [0.8],
                        'colsample_bytree': [0.8]
                    }
                ],
                "catboost": {
                    'iterations': [100, 500, 1000],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 10],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128],
                    'bagging_temperature': [0.0, 1.0, 5.0],
                    'thread_count': [-1],
                }
            }
        
        if pipeline:
            pipeline = Pipeline(preprocessing_pipeline.steps + [('classifier', modelo)])
        else:
            pipeline = modelo

        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})

        # Ajuste del modelo
        grid_search = GridSearchCV(estimator=pipeline, 
                                   param_grid=param_grid, 
                                   cv=cross_validation, 
                                   scoring=score,
                                   verbose=1,
                                   n_jobs=-1)
        

        grid_search.fit(self.X_train, self.y_train)

        best_index = grid_search.best_index_
        self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_
        self.resultados[modelo_nombre]["mean_fit_time"] = grid_search.cv_results_['mean_fit_time'][best_index]
        self.resultados[modelo_nombre]["mean_score_time"] = grid_search.cv_results_['mean_score_time'][best_index]
        self.resultados[modelo_nombre]["best_score"] = grid_search.best_score_
        self.resultados[modelo_nombre]["best_params"] = grid_search.best_params_
        self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
        self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)
        self.resultados[modelo_nombre]["pred_test_prob"] = grid_search.best_estimator_.predict_proba(self.X_test)[:,1]


        # Guardar el modelo
        path = f"../results/{self.iteracion}/{modelo_nombre}"
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/mejor_modelo.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
    

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


        if hasattr(modelo, "predict_proba"):
            prob_train = modelo.predict_proba(self.X_train)[:, 1]
            prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = prob_test = None
        

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
            "average_precision": average_precision_score(self.y_train, prob_train, average='weighted') if prob_train is not None else None,
            "model_mean_fit_time": self.resultados[modelo_nombre]["mean_fit_time"],
            "model_mean_score_time": self.resultados[modelo_nombre]["mean_score_time"],
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
            "average_precision": average_precision_score(self.y_test, prob_test, average='weighted') if prob_test is not None else None,
            "model_mean_fit_time": self.resultados[modelo_nombre]["mean_fit_time"],
            "model_mean_score_time": self.resultados[modelo_nombre]["mean_score_time"],
            "n_jobs": num_nucleos
        }


        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test}).T
    
    def plot_cohens_kappa(self, model_names):
        thresholds = [i / 100 for i in range(1, 100)]

        kappas_dict = {model_name: list() for model_name in model_names}

        for model_name in model_names:
            for threshold in thresholds:
                preds = (self.resultados[model_name]["pred_test_prob"] >= threshold).astype(int)
                kappas_dict[model_name].append(cohen_kappa_score(self.y_test, preds))

            plt.plot(thresholds, kappas_dict[model_name], label=model_name)

        plt.xlabel('Threshold')
        plt.ylabel('Cohen\'s Kappa')
        plt.legend()
        plt.title('Kappa vs Threshold')
        plt.show()

    def plot_matriz_confusion(self, modelo_nombre):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo_nombre})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()


    def importancia_predictores(self, modelo_nombre, pipeline=False):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        # si el modelo está en pipeline, acceder a nombres y algoritmo clasificador
        if pipeline:
            feature_names = modelo[:-1].get_feature_names_out()
            modelo = modelo["classifier"]
        else:
            feature_names = self.X.columns

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
            "Feature":  feature_names,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.show()

    def plot_shap_summary(self, modelo_nombre, pipeline=False):
        """
        Genera un SHAP summary plot para el modelo seleccionado.
        Maneja correctamente modelos de clasificación con múltiples clases.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        # si el modelo está en pipeline, acceder a nombres y algoritmo clasificador
        if pipeline:
            X_test = modelo[:-1].transform(self.X_test)
            feature_names = modelo[:-1].get_feature_names_out()
            modelo = modelo["classifier"]
            
        else:
            X_test = self.X_test
            feature_names = self.X.columns


        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de generar el SHAP plot.")

        # Usar TreeExplainer para modelos basados en árboles
        if modelo_nombre in ["decision_tree", "random_forest", "gradient_boosting", "xgboost", "catboost"]:
            explainer = shap.TreeExplainer(modelo)

            shap_values = explainer.shap_values(X_test)

            # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
            if isinstance(shap_values, list):
                # Para modelos binarios, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Para Decision Trees, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[:, :, 1]
        else:
            # Usar el explicador genérico para otros modelos
            explainer = shap.Explainer(modelo, X_test, check_additivity=False)
            shap_values = explainer(X_test).values

        # Generar el summary plot estándar
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Función para asignar colores
def color_filas_por_modelo(row):
    if row["model"] == "decision_tree":
        return ["background-color: #e6b3e0; color: black"] * len(row)  
    
    elif row["model"] == "random_forest":
        return ["background-color: #c2f0c2; color: black"] * len(row) 

    elif row["model"] == "catboost":
        return ["background-color: #ffd9b3; color: black"] * len(row)  

    elif row["model"] == "xgboost":
        return ["background-color: #f7b3c2; color: black"] * len(row)  

    elif row["model"] == "logistic_regression":
        return ["background-color: #b3d1ff; color: black"] * len(row)  
    
    return ["color: black"] * len(row)
