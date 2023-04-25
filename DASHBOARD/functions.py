from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import STLForecast
from pmdarima.arima import auto_arima
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance, plot_tree
import xgboost as xgb
from prophet import Prophet
import pandas as pd
import numpy as np
import seaborn as sns
import math
from datetime import date, time, datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hdbcli import dbapi as db
from statsmodels.tsa.api import STLForecast, ExponentialSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score
from statsmodels.tsa.api import STLForecast, ExponentialSmoothing, ETSModel
import warnings
warnings.filterwarnings('ignore')


def load_data():
    conn = db.connect(address='10.0.0.110', port='30015',
                      user='DESARROLLO', password='DevHermeco.2022')
    sql = '''SELECT "AÑO",
    "MES",
    "UNIDADES",
    "Sublínea",
    "Mundo",
    "Grupo de Artículo",
    "Tipo de Artículo",
    "Concepto Diseño",
    "FECHA VENTA",
    "Organización Ventas"
    FROM HEP300.VW_CD_REPORTE_VENTAS_ALL_2016 
    WHERE (CANAL = 'FRANQUICIAS' OR CANAL = 'TIENDAS PROPIAS') AND "AÑO" >= 2018
    '''
    df_canal = pd.read_sql_query(sql, conn)
    df_canal = df_canal.loc[df_canal['Sublínea'] == 'BASICOS']

    return df_canal


def longitud_data(mundo, tipo, df):
    df_filtrado = df.loc[(df['Mundo'] == mundo) &
                         (df["Tipo de Artículo"] == tipo)]

    if (len(df_filtrado) == 0):
        longitud_mes = 0
    else:
        longitud_mes = len(df_filtrado)

    return longitud_mes


def show_Data(mundo, tipo, df):
    # Esta funcion recibe 2 parametros, MUNDO Y TIPO DE ARTICULO
    # sobre el cual se desea hacer el pronostico.

    # la funcion retorna un dataframe con los elementos de entrada
    # contiene las columnas AÑO DE VENTA,
    #                       MES DE VENTA
    #                       MES
    #                       AÑO
    #                       UNIDADES

    data = df.loc[df["Mundo"] == mundo]
    data = data.loc[data["Tipo de Artículo"] == tipo]

    data_df = data.groupby(['MES', 'AÑO'])[
        'UNIDADES'].sum().to_frame().sort_values(by=['AÑO', 'MES']).reset_index()

    data_df['MES_AÑO'] = pd.to_datetime(data_df['AÑO'].astype(
        str) + '-' + data_df['MES'].astype(str)).dt.strftime('%Y-%m')

    return data_df


def calcular_Cv(basicos1):
    meses = basicos1.MES.unique()
    # meses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Calcular el factor CV para cada mes
    cv_mes = {}
    for mes in meses:
        if mes in basicos1['MES'].unique():
            cv_mes[mes] = round(basicos1[basicos1['MES'] == mes]['UNIDADES'].std()
                                / basicos1[basicos1['MES'] == mes]['UNIDADES'].mean(), 2)
        # else:
        #     cv_mes[mes] = 0

    lista_cv = []
    for i in cv_mes.values():
        lista_cv.append(i)

    coeficiente_Var = np.std(lista_cv)/np.mean(lista_cv)

    return round(coeficiente_Var, 2)


# FUNCIONES MODELO
# 1. Preparacion de los datos


def DATA_preparation_2022(mundo, tipo, df):
    # Esta funcion recibe 2 parametros, MUNDO Y TIPO DE ARTICULO
    # sobre el cual se desea hacer el pronostico.

    # la funcion retorna un dataframe con los elementos de entrada
    # contiene las columnas AÑO DE VENTA,
    #                       MES DE VENTA
    #                       MES
    #                       AÑO
    #                       UNIDADES

    data = df.loc[df["Mundo"] == mundo]
    data = data.loc[data["Tipo de Artículo"] == tipo]

    data_df = data.groupby(['MES', 'AÑO'])[
        'UNIDADES'].sum().to_frame().sort_values(by=['AÑO', 'MES']).reset_index()

    # filtrar los meses de interes OCT-MAR
    df_model = data_df.loc[(data_df['MES'] >= 9) | (data_df['MES'] < 3)]

    # CREAR COLUMNA AÑO Y MES SEGUN MES Y AÑO DE VENTA
    df_model['MES_AÑO'] = pd.to_datetime(df_model['AÑO'].astype(
        str) + '-' + df_model['MES'].astype(str)).dt.strftime('%Y-%m')

    return df_model.reset_index()

# 2. Definicion caracteristicas para el modelo
# por ahora son solo temporales


def create_features_mes_venta(df, label=None):
    df['date'] = pd.to_datetime(df.index)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['covid'] = [1 if ((x > pd.Timestamp('2020-03')) &
                         (x < pd.Timestamp('2021-03'))) else 0 for x in df.date]

    X = df[['quarter', 'month', 'year',
           'dayofyear', 'weekofyear', 'covid']]

    if label:
        y = df[label]
        return X, y
    return X

# 3. Implementacion del modelo


def XGB_model_2022(df):
    #  TRAIN & TEST
    train = df.loc[df.MES_AÑO <= '2022-03'].set_index('MES_AÑO')
    test = df.loc[(df.MES_AÑO >= '2022-09') &
                  (df.MES_AÑO <= '2023-02')].set_index('MES_AÑO')

    # Define el train y test de acuerdo con las bases de datos de train y de test
    X_train, y_train = create_features_mes_venta(train, label='UNIDADES')
    X_test, y_test = create_features_mes_venta(test, label='UNIDADES')

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=0.01,
                           eval_metric=['rmse', 'mae', 'mape']
                           )

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False)

    pred_test = reg.predict(X_test)
    test['forecast'] = np.ceil(pred_test)

    df_xgb = pd.concat([train['UNIDADES'].to_frame(),
                        test[['UNIDADES', 'forecast']]]).reset_index()

    return test[['UNIDADES', 'forecast']]

# 4. Funcion que enetrega los resultados del modelo al implementarlo
# mes a mes, es decir, se agrupa la informacion relevante a cada mes
# Ejemplo, tomamos todos los meses de enero implicados en la serie y se pronostica el siguiente
# mes de eneo, y así suacesivamente con todos los datos


def resultados_modelo_mensual(df):
    meses_interes = [1, 2, 9, 10, 11, 12]
    basicos_forecast = {}

    basicos_results = pd.DataFrame()

    for mes in meses_interes:
        if mes in df['MES'].unique():
            # basicos_forecast[mes] = XGB_model_2022(basicos1[basicos1['MES'] == mes])
            basicos_results = pd.concat(
                [basicos_results, XGB_model_2022(df[df['MES'] == mes])])

        else:
            basicos_forecast[mes] = 0
