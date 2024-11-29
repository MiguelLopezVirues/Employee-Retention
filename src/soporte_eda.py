# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np


# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
import math
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest

# Visualizacion de datos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic


def exploracion_dataframe(dataframe):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Las primeras filas del dataframe son:")
    display(dataframe.head(2))
    print("\n ..................... \n")

    print(f"Los tipos de las columnas y sus valores únicos son:")
    datos_conteos = pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"])
    datos_conteos["conteo"] = dataframe.nunique()
    display(datos_conteos)
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] >0])
    print("\n ..................... \n")



    # controlar valores únicos de las variables
    print("Comprobamos una representación mínima para valores numéricos:")
    for feature in dataframe.columns:
        if dataframe[feature].nunique() == 1:
            print(f"● La variable {feature} tiene 1 solo valor único. Se elimina.")
            dataframe.drop(columns=feature, inplace=True)
    print("\n ..................... \n")
    

    # controlar valores únicos de las variables numericas
    print("Comprobamos una representación mínima para valores numéricos:")
    for feature in dataframe.select_dtypes(np.number).columns:
        if dataframe[feature].nunique() <= 15:
            print(f"● La variable {feature} tiene {dataframe[feature].nunique()} < 15 valores únicos. Se convierte a objeto.")
            dataframe[feature] = dataframe[feature].astype("object")

    print("\n ..................... \n")
        

    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene {dataframe[col].nunique()} valores únicos, de los cuales los primeros son:")
        display(pd.DataFrame(dataframe[col].value_counts()).assign(pct=lambda x: round(x["count"]/dataframe.shape[0],3)*100).head())    
    



def custom_properties(mapping):
    def properties(key):
        purchase_status = str(key[1])  # Extract PurchaseStatus from key
        return {"color": mapping.get(purchase_status, "gray")}  # Default to gray if not mapped
    return properties

def plot_relationships_categorical_target(df, target,hue=None, cat_type="count", num_type="hist",mapping={}):
    columns = df.drop(columns=target).columns.to_list()

    num_cols = 2
    num_rows = math.ceil(len(columns)/num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15,num_rows*4))
    axes = axes.flat

    fig.suptitle("Difference in distrubtion by target class.", y=0.93)
    for ax, feature in zip(axes, columns):
        if df[feature].dtype in [int,float]:
            if num_type == "box":
                sns.boxplot(data=df,
                            x=target,
                            y=feature,
                            ax=ax,
                            hue=None)
            else:
                sns.histplot(data=df,
                                x=feature,
                                hue=target,
                                ax=ax,
                                stat="proportion")

        else:
            # mosaic plots
            if cat_type == "mosaic":
                mosaic(df, [feature,target], properties=custom_properties(mapping), ax=ax)
            else:
                sns.countplot(data=df,
                            x=feature,
                            hue=target,
                            ax=ax)

            
            ax.set_title(feature)

    if len(columns) % 2 != 0:
            fig.delaxes(ax=axes[-1])


    plt.subplots_adjust(hspace=0.6)
    plt.show()