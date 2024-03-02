import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np


# Function to compute response r(t) and update accumulated value G(t)
def compute_response(L_t, K, sigma, alpha, G_prev):
    G_prev = min(G_prev, K)
    r_t = max(L_t * ((K - G_prev) ** 0.5) / sigma, 0)
    G_prev = (1 - alpha) * G_prev + alpha * r_t
    return r_t, G_prev


# Original input data
L_t_series = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
colors = ['blue', 'green', 'red']

# Initialize the app and use a bootstrap style
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the web app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Interactive Layer Parameters"), width={'size': 6, 'offset': 3}), align='center'),

    dbc.Row(dbc.Col(dcc.Graph(id='graph'), width=12), align='center'),

    # Sliders for parameters
    dbc.Row([
        dbc.Col([
            html.H6("Layer 1 Parameters:"),
            html.Label("K:"),
            dcc.Slider(id='K1-slider', min=-1, max=4, step=0.1, value=2.),
            html.Label("alpha:"),
            dcc.Slider(id='alpha1-slider', min=-1, max=4, step=0.1, value=0.1),
            html.Label("sigma:"),
            dcc.Slider(id='sigma1-slider', min=-1, max=4, step=0.1, value=1.),
        ], width=4),
        dbc.Col([
            html.H6("Layer 2 Parameters:"),
            html.Label("K:"),
            dcc.Slider(id='K2-slider', min=-1, max=4, step=0.1, value=2.),
            html.Label("alpha:"),
            dcc.Slider(id='alpha2-slider', min=-1, max=4, step=0.1, value=.1),
            html.Label("sigma:"),
            dcc.Slider(id='sigma2-slider', min=-1, max=4, step=0.1, value=1),
        ], width=4),
        dbc.Col([
            html.H6("Layer 3 Parameters:"),
            html.Label("K:"),
            dcc.Slider(id='K3-slider', min=-1, max=4, step=0.1, value=2.),
            html.Label("alpha:"),
            dcc.Slider(id='alpha3-slider', min=-1, max=4, step=0.1, value=.1),
            html.Label("sigma:"),
            dcc.Slider(id='sigma3-slider', min=-1, max=4, step=0.1, value=1.),
        ], width=4)
    ]),
], fluid=True)


# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [
        Input('K1-slider', 'value'), Input('alpha1-slider', 'value'), Input('sigma1-slider', 'value'),
        Input('K2-slider', 'value'), Input('alpha2-slider', 'value'), Input('sigma2-slider', 'value'),
        Input('K3-slider', 'value'), Input('alpha3-slider', 'value'), Input('sigma3-slider', 'value'),
    ]
)
def update_figure(K1, alpha1, sigma1, K2, alpha2, sigma2, K3, alpha3, sigma3):
    params = [
        {"K": K1, "alpha": alpha1, "sigma": sigma1},
        {"K": K2, "alpha": alpha2, "sigma": sigma2},
        {"K": K3, "alpha": alpha3, "sigma": sigma3}
    ]

    fig = go.Figure()
    input_series = L_t_series

    for i, param in enumerate(params):
        r_t_series, G_t_series = [], []
        G_prev = 0.0

        for L_t in input_series:
            r_t, G_prev = compute_response(L_t, param["K"], param["sigma"], param["alpha"], G_prev)
            r_t_series.append(r_t)
            G_t_series.append(G_prev)

        fig.add_trace(
            go.Scatter(y=r_t_series, mode='lines', name=f'Layer {i + 1} Response $r(t)$', line=dict(color=colors[i])))
        fig.add_trace(go.Scatter(y=G_t_series, mode='lines', name=f'Layer {i + 1} Accumulated Value $G(t)$',
                                 line=dict(color=colors[i], dash='dash')))

        input_series = r_t_series

    fig.add_trace(
        go.Scatter(y=L_t_series, mode='lines+markers', name='Input $L(t)$', line=dict(color='purple', dash='dot')))
    fig.update_layout(
        title='Responses $r(t)$, Input $L(t)$, and Accumulated Values $G(t)$ over Time',
        xaxis_title='Time $t$',
        yaxis_title='Value',
        showlegend=True
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
