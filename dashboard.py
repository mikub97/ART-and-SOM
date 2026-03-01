import dash
from dash import dcc, html

import ART1
import ART2A_E
import SOM

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Neural Learning Algorithms Dashboard",
            style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif',
                   'margin': '10px 0', 'fontSize': '24px'}),
    dcc.Tabs(children=[
        dcc.Tab(label='ART1',    children=[ART1.layout]),
        dcc.Tab(label='ART2A-E', children=[ART2A_E.layout]),
        dcc.Tab(label='SOM',     children=[SOM.layout]),
    ])
])

ART1.register_callbacks(app)
ART2A_E.register_callbacks(app)
SOM.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
