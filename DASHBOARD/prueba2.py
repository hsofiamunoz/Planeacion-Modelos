import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(
    children=[
        html.Div(
            children="Date Range", className="menu-title"
        ),
        dcc.DatePickerRange(
            id="date-range",
            start_date_placeholder_text="Start Date",
            end_date_placeholder_text="End Date",
            calendar_orientation="vertical",
        ),
    ]
)


@app.callback(
    Output("output", "children"),
    [Input("date-range", "start_date"), Input("date-range", "end_date")]
)
def update_output(start_date, end_date):
    return f"You selected {start_date} to {end_date}"


if __name__ == "__main__":
    app.run_server(debug=True)
