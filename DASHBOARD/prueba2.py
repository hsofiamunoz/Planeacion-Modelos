
import pandas as pd
from dash import Dash, dcc, html
import functions

root = 'C:/Users/HELENAMM/OneDrive - offcorss.com/OC Magic Data/Proyecto pronostico/Modelos fase 2/Data/VENTAS_2018_1.xlsx'
data = (
    pd.read_excel('prueba_dash.xlsx')
    # .query("type == 'conventional' and region == 'Albany'")
    .assign(Date=lambda data: pd.to_datetime(data["FECHA VENTA"], format="%Y-%m-%d"))
    .sort_values(by="Date")
)

# data = pd.read_excel(root).assign(Date=lambda data: pd.to_datetime(
#     data["FECHA VENTA"], format="%Y-%m-%d")).sort_values(by="Date")

data_plot = (functions.DATA_preparation_2022('NIÑA', 'MINIFALDA', data))

app = Dash(__name__)


app.layout = html.Div(
    children=[
        html.H1(children="Avocado Analytics"),
        html.P(
            children=(
                "Analyze the behavior of avocado prices and the number"
                " of avocados sold in the US between 2015 and 2018"
            ),
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data_plot["MES_AÑO"],
                        "y": data_plot["UNIDADES"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Average Price of Avocados"},
            },
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data["Date"],
                        "y": data["UNIDADES"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Avocados Sold"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
