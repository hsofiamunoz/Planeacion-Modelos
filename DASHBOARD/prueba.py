# Fuente
# https://realpython.com/python-dash/#what-is-dash

# -----------------------------------------------------------------------
# LIBRERIAS
import pandas as pd
from dash import Dash, dcc, html, dash_table
import functions
from dash.dependencies import Input, Output

# -----------------------------------------------------------------------
# DATOS
data = functions.load_data()

mundos = data["Mundo"].sort_values().unique()
tipos_articulo = sorted(data["Tipo de Artículo"].unique())
canales = ['TIENDAS PROPIAS', 'FRANQUICIAS',
           'CADENAS', 'VENTA DIRECTA', 'TIENDA VIRTUAL']

concepto_diseno = data["Concepto Diseño"].sort_values().unique()


data_plot = functions.DATA_preparation_2022('NIÑO', 'CAMISA MANGA LARGA', data)

# -----------------------------------------------------------------------
# APLICACION

app = Dash(__name__)
app.title = "Piloto 1"
# app._favicon = "assets/logo.ico"

app.layout = html.Div(
    children=[
        # HEADER
        html.Div(
            children=(
                html.P(children="🦁",
                       className="header-emoji"),
                html.H1(
                    children="MODELO PRONOSTICO",
                    className="header-title"
                ),
                html.P(
                    children=(
                        "Sistema pronosticos para SPECIAL OCCASIONS"
                        " - Ciencia de Datos"
                    ),
                    className="header-description"
                )
            ), className='header'
        ),

        # MENU
        html.Div(
            children=[
                # MUNDO
                html.Div(
                    children=[
                        html.Div(children="Mundo", className="menu-title"),
                        dcc.Dropdown(
                            id="mundo-filter",
                            options=[
                                {"label": mundo, "value": mundo}
                                for mundo in mundos
                            ],
                            value="NIÑO",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                # TIPO ARTICULO
                html.Div(
                    children=[
                        html.Div(children="Tipo", className="menu-title"),
                        dcc.Dropdown(
                            id="tipo-filter",
                            options=[
                                {"label": tipo, "value": tipo}
                                for tipo in tipos_articulo
                            ],
                            value="CAMISA MANGA LARGA",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),

                # Rango de fechas
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            # min_date_allowed=data["FECHA VENTA"].min().date(),
                            # max_date_allowed=data["FECHA VENTA"].max().date(),
                            # start_date=data["FECHA VENTA"].min().date(),
                            # end_date=data["FECHA VENTA"].max().date(),
                        ),
                    ]
                ),

            ],
            className="menu",
        ),

        html.Div(children=[html.H1(children="")]),
        # MENU 2
        html.Div(
            children=[
                # CANALES
                html.Div(
                    children=[
                        html.Div(children="Canal", className="menu-title"),
                        dcc.Dropdown(
                            id="canal-filter",
                            options=[
                                {"label": canal, "value": canal}
                                for canal in canales
                            ],
                            value="CANAL",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                # CONCEPTO
                html.Div(
                    children=[
                        html.Div(children="Concepto Diseño",
                                 className="menu-title"),
                        dcc.Dropdown(
                            id="concepto-filter",
                            options=[
                                {"label": concepto, "value": concepto}
                                for concepto in concepto_diseno
                            ],
                            value="BASICOS",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                # TIPO ARTICULO
                # html.Div(
                #     children=[
                #         html.Div(children="Tipo", className="menu-title"),
                #         dcc.Dropdown(
                #             id="tipo-filter",
                #             options=[
                #                 {"label": tipo, "value": tipo}
                #                 for tipo in tipos_articulo
                #             ],
                #             value="CAMISA MANGA LARGA",
                #             clearable=False,
                #             searchable=False,
                #             className="dropdown",
                #         ),
                #     ],
                # ),

            ],
            className="menu",
        ),


        html.Div(id='output'),
        html.Div(children=html.H1("")),
        html.Div(id='output2'),

        # GRAFICAS
        html.Div(
            children=[
                # GRAFICA HISTORICO
                html.Div(
                    children=dcc.Graph(
                        id="price-chart",
                        config={"displayModeBar": False},
                        figure={
                            "data": [
                                {
                                    "x": data_plot['MES_AÑO'],
                                    "y": data_plot['UNIDADES'],

                                    "type": "lines",
                                },
                            ],
                            "layout": {
                                "title": {
                                    "text": "Serie temporal",
                                    "x": 0.05,
                                    "xanchor": "left",
                                },
                                "xaxis": {"fixedrange": True},
                                "yaxis": {
                                    # "tickprefix": "$",
                                    "fixedrange": True,
                                },
                                "colorway": ["#FFA500"],  # D68910  ##17202A
                            },
                        },
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),

        # BOTON
        html.Div(
            children=[
                html.Button('Generar Pronostico', id='my-button')
            ]
        ),
        html.Div(children=html.H1("")),
        html.Div(id='mensaje_boton')

    ]
)

# CALLBACK TEXTO CON CANTIDAD DE INFORMACION


@ app.callback(
    Output('output', component_property='children'),
    Input("mundo-filter", component_property="value"),
    Input("tipo-filter", component_property="value")
)
def update_output(mundo, tipo):
    if (functions.longitud_data(mundo, tipo, data) == 0):
        return "No hay información Disponible"
    else:
        mes_total = len(functions.DATA_preparation_2022(
            mundo, tipo, data))
        return f"Usted ha seleccionado {mundo} and {tipo}. Cuenta con {mes_total} datos "


@ app.callback(
    Output('output2', component_property='children'),
    Input("mundo-filter", component_property="value"),
    Input("tipo-filter", component_property="value")
)
def update_output2(mundo, tipo):
    mes_total = len(functions.DATA_preparation_2022(
        mundo, tipo, data))

    if (mes_total < 12):
        return "No hay informacion suficiente"
    else:
        basicos1 = functions.DATA_preparation_2022(mundo, tipo, data)
        cv = functions.calcular_Cv(basicos1)
        return f"Esta serie tiene un coeficiente de : {str(cv)}"


# CALLBACK - ACTUALIZACION GRAFICAS HISTORICO

@ app.callback(
    Output('price-chart', component_property='figure'),
    Input("mundo-filter", component_property="value"),
    Input("tipo-filter", component_property="value")
)
def update_charts(mundo, tipo):
    filtered_data = functions.show_Data(mundo, tipo, data)

    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["MES_AÑO"],
                "y": filtered_data["UNIDADES"],
                "type": "lines",
                # "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Grafica de ventas",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            # "yaxis": {"tickprefix": "$", "fixedrange": True},
            "colorway": ["#FFA500"],
        },
    }
    return price_chart_figure


@app.callback(
    Output('mensaje_boton', 'children'),
    Input('my-button', 'n_clicks')
)
def mensaje_boton(n_clicks):
    if n_clicks is not None:
        return f"hola mundo"
# def generate_table(n_clicks):
#     if n_clicks is not None:
#         # Generate table here based on the button click
#         table_data = [['Name', 'Age'], ['John', '28'], ['Jane', '25']]
#         rows = []
#         for row in table_data:
#             rows.append(html.Tr([html.Td(cell) for cell in row]))
#         return rows


# INICIALIZACION DEL TABLERO
if __name__ == "__main__":
    app.run_server(debug=True)
