import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import load_iris

# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------
iris = load_iris()
X_raw = iris.data[:, :2]
y_iris = iris.target
target_names = iris.target_names

# Standardize to [0.1, 0.9]
X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
X_std = 0.1 + 0.8 * (X_raw - X_min) / (X_max - X_min)

# -----------------------------------------------------------------------------
# DYNAMIC CODE STRINGS
# -----------------------------------------------------------------------------
som_code_1d = """class SOM_1D:
    def __init__(self, n_neurons, input_dim=2):
        self.codebook = np.random.random((n_neurons, input_dim))

    def learn(self, samples, sigma, lrate):
        # 1. LOAD SAMPLE
        data = samples[np.random.randint(0, len(samples))]

        # 2. FIND BMU (1D Index)
        D = ((self.codebook - data)**2).sum(axis=-1)
        winner = np.argmin(D)

        # 3. CALC NEIGHBORHOOD (1D Distance)
        dist_sq = np.abs(np.arange(self.codebook.shape[0]) - winner)**2
        G = np.exp(-dist_sq / (2 * sigma**2))

        # 4. UPDATE WEIGHTS
        delta = data - self.codebook
        self.codebook += lrate * G[:, None] * delta"""

som_code_2d = """class SOM_2D:
    def __init__(self, rows, cols, input_dim=2):
        self.codebook = np.random.random((rows, cols, input_dim))

    def learn(self, samples, sigma, lrate):
        # 1. LOAD SAMPLE
        data = samples[np.random.randint(0, len(samples))]

        # 2. FIND BMU (2D Coordinates)
        D = ((self.codebook - data)**2).sum(axis=-1)
        winner = np.unravel_index(np.argmin(D), D.shape)

        # 3. CALC NEIGHBORHOOD (2D Grid Distance)
        r, c = np.ogrid[:self.codebook.shape[0], :self.codebook.shape[1]]
        dist_sq = (r - winner[0])**2 + (c - winner[1])**2
        G = np.exp(-dist_sq / (2 * sigma**2))

        # 4. UPDATE WEIGHTS
        delta = data - self.codebook
        self.codebook += lrate * G[..., None] * delta"""

# -----------------------------------------------------------------------------
# APP STATE
# -----------------------------------------------------------------------------
class FullSOMState:
    def __init__(self):
        self.reset(n=3, topology="1D")

    def reset(self, n=3, topology="1D"):
        self.n_neurons = n
        self.topology = topology
        if topology == "1D":
            self.codebook = np.random.random((n, 2))
        else:
            self.grid_rows = int(np.sqrt(n))
            self.grid_cols = max(1, int(np.round(n / self.grid_rows)))
            self.n_neurons = self.grid_rows * self.grid_cols
            self.codebook = np.random.random((self.grid_rows, self.grid_cols, 2))

        self.current_sample = np.zeros(2)
        self.winner = None
        self.phase = "LOAD_SAMPLE"
        self.status_log = f"Reset: {topology} with {self.n_neurons} neurons.\n"
        self.play_mode = False
        self.sample_counter = 0
        self.errors = []

som_state = FullSOMState()


# -----------------------------------------------------------------------------
# U-MATRIX HELPERS
# -----------------------------------------------------------------------------
def calculate_umatrix(codebook):
    """Average distance to 4-connected neighbours for each neuron in a 2D grid."""
    rows, cols, _ = codebook.shape
    u_matrix = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            neighbors = []
            if r > 0:      neighbors.append(np.linalg.norm(codebook[r, c] - codebook[r-1, c]))
            if r < rows-1: neighbors.append(np.linalg.norm(codebook[r, c] - codebook[r+1, c]))
            if c > 0:      neighbors.append(np.linalg.norm(codebook[r, c] - codebook[r, c-1]))
            if c < cols-1: neighbors.append(np.linalg.norm(codebook[r, c] - codebook[r, c+1]))
            if neighbors:
                u_matrix[r, c] = np.mean(neighbors)
    return u_matrix


def build_umatrix_figure():
    if som_state.topology != "2D":
        fig_u = go.Figure()
        fig_u.add_annotation(
            text="U-Matrix is only available for 2D topology",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color='grey')
        )
        fig_u.update_layout(
            title="U-Matrix", height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis={'visible': False}, yaxis={'visible': False}
        )
        return fig_u

    u_matrix = calculate_umatrix(som_state.codebook)

    fig_u = go.Figure()
    fig_u.add_trace(go.Heatmap(
        z=u_matrix, colorscale='Viridis',
        colorbar=dict(title='Mean Dist', thickness=12, len=0.8),
        showscale=True, name='U-Matrix'
    ))

    # Overlay BMU positions per Iris class with stable jitter
    rng = np.random.default_rng(42)
    s_colors  = ['cyan', 'lime', 'magenta']
    s_markers = ['circle', 'square', 'triangle-up']
    for cls in range(3):
        xs, ys = [], []
        for i, x in enumerate(X_std):
            if y_iris[i] == cls:
                D = ((som_state.codebook - x) ** 2).sum(axis=-1)
                bmu_r, bmu_c = np.unravel_index(np.argmin(D), D.shape)
                xs.append(bmu_c + rng.uniform(-0.25, 0.25))
                ys.append(bmu_r + rng.uniform(-0.25, 0.25))
        fig_u.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            marker=dict(color=s_colors[cls], symbol=s_markers[cls], size=8,
                        line=dict(color='black', width=1)),
            name=target_names[cls]
        ))

    fig_u.update_layout(
        title="U-Matrix: Iris Topology Map",
        xaxis_title="Grid Column",
        yaxis_title="Grid Row",
        height=320,
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(orientation="h", y=1.12)
    )
    return fig_u


def generate_code_display(phase, locked, topology):
    if topology == "1D":
        lines = som_code_1d.split('\n')
        h = []
        if not locked:
            if phase == "LOAD_SAMPLE":         h = [5, 6]
            elif phase == "FIND_BMU":          h = [8, 9, 10]
            elif phase == "CALC_NEIGHBORHOOD": h = [12, 13, 14]
            elif phase == "UPDATE_WEIGHTS":    h = [16, 17, 18]
    else:
        lines = som_code_2d.split('\n')
        h = []
        if not locked:
            if phase == "LOAD_SAMPLE":         h = [5, 6]
            elif phase == "FIND_BMU":          h = [8, 9, 10]
            elif phase == "CALC_NEIGHBORHOOD": h = [12, 13, 14, 15]
            elif phase == "UPDATE_WEIGHTS":    h = [17, 18, 19]

    return [html.Div(line, style={
        'paddingLeft': '10px', 'whiteSpace': 'pre', 'fontFamily': 'monospace', 'fontSize': '12px',
        'backgroundColor': '#e2f0d9' if i in h else 'transparent',
        'fontWeight': 'bold' if i in h else 'normal',
        'borderLeft': '4px solid #70ad47' if i in h else '4px solid transparent'
    }) for i, line in enumerate(lines)]

# -----------------------------------------------------------------------------
# LAYOUT (importable)
# -----------------------------------------------------------------------------
layout = html.Div(
    style={'display': 'flex', 'padding': '20px', 'gap': '20px', 'height': '85vh',
           'fontFamily': 'Segoe UI, Arial'},
    children=[
        dcc.Interval(id='som-play-timer', interval=200, disabled=True),

        html.Div(style={'flex': '0 0 400px', 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
            html.H3("Status Log"),
            dcc.Textarea(id='som-log', readOnly=True,
                         style={'width': '100%', 'height': '120px', 'fontSize': '11px',
                                'fontFamily': 'monospace', 'backgroundColor': '#fff'}),
            html.H3("Algorithm Source View"),
            html.Div(id='som-code-panel', style={
                'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'flex': '1',
                'overflowY': 'auto', 'padding': '10px'
            })
        ]),

        html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}, children=[
            html.H2("SOM 2D Dashboard"),

            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px',
                            'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px',
                            'border': '1px solid #eee'}, children=[
                html.Div(style={'display': 'flex', 'gap': '30px', 'alignItems': 'center'}, children=[
                    html.Div([
                        html.B("Topology:"),
                        dcc.RadioItems(id='som-topo-select',
                                       options=[{'label': '1D Chain', 'value': '1D'},
                                                {'label': '2D Grid', 'value': '2D'}],
                                       value='1D', inline=True, style={'marginLeft': '10px'})
                    ]),
                    html.Button("Next Step", id='som-next', n_clicks=0,
                                style={'padding': '8px 15px', 'cursor': 'pointer'}),
                    html.Button("Play Sequence", id='som-play', n_clicks=0,
                                style={'padding': '8px 15px', 'backgroundColor': '#17a2b8',
                                       'color': 'white', 'cursor': 'pointer'}),
                    html.Button("Reset Network", id='som-reset', n_clicks=0,
                                style={'padding': '8px 15px', 'backgroundColor': '#DC3545',
                                       'color': 'white', 'cursor': 'pointer'}),
                ]),

                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '30px'}, children=[
                    html.Div([
                        html.Label("Neuron Count (n):", style={'fontWeight': 'bold'}),
                        dcc.Slider(id='som-n-slider', min=3, max=30, step=1, value=3,
                                   marks={3: '3', 15: '15', 30: '30'},
                                   tooltip={"always_visible": True, "placement": "top"})
                    ]),
                    html.Div([
                        html.Label("Sigma (Neighborhood):", style={'fontWeight': 'bold'}),
                        dcc.Slider(id='som-sigma-s', min=0.1, max=3.0, step=0.1, value=0.6,
                                   marks={0.1: '0.1', 1.5: '1.5', 3.0: '3.0'},
                                   tooltip={"always_visible": True, "placement": "top"})
                    ]),
                    html.Div([
                        html.Label("Speed (ms per step):", style={'fontWeight': 'bold'}),
                        dcc.Slider(id='som-speed-s', min=20, max=1000, step=50, value=150,
                                   marks={20: 'Fast', 1000: 'Slow'},
                                   tooltip={"always_visible": True, "placement": "top"})
                    ])
                ])
            ]),

            html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column',
                            'gap': '10px', 'overflowY': 'auto'}, children=[
                dcc.Graph(id='som-graph',        style={'height': '380px', 'flex': 'none'}),
                dcc.Graph(id='som-error-graph',  style={'height': '180px', 'flex': 'none'}),
                dcc.Graph(id='som-umatrix-graph', style={'height': '340px', 'flex': 'none'})
            ])
        ])
    ]
)

# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------
def register_callbacks(app):
    @app.callback(
        [Output('som-graph', 'figure'), Output('som-error-graph', 'figure'),
         Output('som-log', 'value'), Output('som-code-panel', 'children'),
         Output('som-play-timer', 'disabled'), Output('som-play-timer', 'interval'),
         Output('som-umatrix-graph', 'figure')],
        [Input('som-next', 'n_clicks'), Input('som-reset', 'n_clicks'),
         Input('som-play', 'n_clicks'), Input('som-play-timer', 'n_intervals'),
         Input('som-n-slider', 'value'), Input('som-topo-select', 'value')],
        [State('som-sigma-s', 'value'), State('som-speed-s', 'value')]
    )
    def master_callback(n_next, n_reset, n_play, n_int, n_val, topo_val, sigma, speed):
        tid = ctx.triggered_id
        lrate = 0.5

        if tid in ['som-reset', 'som-n-slider', 'som-topo-select']:
            som_state.reset(n_val, topo_val)

        elif tid == 'som-play':
            som_state.play_mode = not som_state.play_mode
            som_state.status_log += f"Play mode toggled.\n"

        elif tid in ['som-next', 'som-play-timer']:
            if som_state.play_mode:
                idx = som_state.sample_counter % len(X_std)
                data = X_std[idx]
                dists = ((som_state.codebook - data)**2).sum(axis=-1)
                winner = np.unravel_index(np.argmin(dists), dists.shape)

                if som_state.topology == "1D":
                    g = np.array([np.exp(-abs(i - winner[0])**2 / (2 * sigma**2)) for i in range(som_state.n_neurons)])
                    som_state.codebook += lrate * g[:, None] * (data - som_state.codebook)
                else:
                    r, c = np.ogrid[:som_state.codebook.shape[0], :som_state.codebook.shape[1]]
                    d2 = (r - winner[0])**2 + (c - winner[1])**2
                    g = np.exp(-d2 / (2 * sigma**2))
                    som_state.codebook += lrate * g[..., None] * (data - som_state.codebook)

                som_state.current_sample = data
                som_state.winner = winner
                som_state.sample_counter += 1
                som_state.errors.append(np.min(dists))
            else:
                if som_state.phase == "LOAD_SAMPLE":
                    som_state.current_sample = X_std[np.random.randint(0, len(X_std))]
                    som_state.status_log += "1. Sample Loaded.\n"
                    som_state.phase = "FIND_BMU"
                elif som_state.phase == "FIND_BMU":
                    dists = ((som_state.codebook - som_state.current_sample)**2).sum(axis=-1)
                    som_state.winner = np.unravel_index(np.argmin(dists), dists.shape)
                    som_state.errors.append(np.min(dists))
                    som_state.status_log += f"2. BMU Winner: {som_state.winner}\n"
                    som_state.phase = "CALC_NEIGHBORHOOD"
                elif som_state.phase == "CALC_NEIGHBORHOOD":
                    som_state.status_log += "3. Gaussian Bubble Calculated.\n"
                    som_state.phase = "UPDATE_WEIGHTS"
                elif som_state.phase == "UPDATE_WEIGHTS":
                    winner = som_state.winner
                    data = som_state.current_sample
                    if som_state.topology == "1D":
                        g = np.array([np.exp(-abs(i - winner[0])**2 / (2 * sigma**2)) for i in range(som_state.n_neurons)])
                        som_state.codebook += lrate * g[:, None] * (data - som_state.codebook)
                    else:
                        r, c = np.ogrid[:som_state.codebook.shape[0], :som_state.codebook.shape[1]]
                        d2 = (r - winner[0])**2 + (c - winner[1])**2
                        g = np.exp(-d2 / (2 * sigma**2))
                        som_state.codebook += lrate * g[..., None] * (data - som_state.codebook)
                    som_state.status_log += "4. NEURONS MOVED!\n"
                    som_state.phase = "LOAD_SAMPLE"

        # BUILD MAP FIGURE
        fig = go.Figure()
        s_colors = ['#636EFA', '#00CC96', '#EF553B']

        for i, name in enumerate(target_names):
            mask = y_iris == i
            fig.add_trace(go.Scatter(x=X_std[mask, 0], y=X_std[mask, 1], mode='markers',
                                     marker=dict(color=s_colors[i], size=6, opacity=0.3), name=name))

        if not np.all(som_state.current_sample == 0):
            fig.add_trace(go.Scatter(x=[som_state.current_sample[0]], y=[som_state.current_sample[1]],
                                     mode='markers',
                                     marker=dict(color='red', size=22, symbol='star',
                                                 line=dict(color='black', width=1)), name='Target'))

        if som_state.topology == "1D":
            fig.add_trace(go.Scatter(x=som_state.codebook[:,0], y=som_state.codebook[:,1],
                                     mode='lines+markers',
                                     marker=dict(color='black', size=8),
                                     line=dict(color='black', width=1, dash='dot'), showlegend=False))
        else:
            rows, cols = som_state.codebook.shape[0], som_state.codebook.shape[1]
            for r in range(rows):
                for c in range(cols):
                    if c < cols - 1:
                        fig.add_trace(go.Scatter(x=[som_state.codebook[r,c,0], som_state.codebook[r,c+1,0]],
                                                 y=[som_state.codebook[r,c,1], som_state.codebook[r,c+1,1]],
                                                 mode='lines', line=dict(color='black', width=1), showlegend=False))
                    if r < rows - 1:
                        fig.add_trace(go.Scatter(x=[som_state.codebook[r,c,0], som_state.codebook[r+1,c,0]],
                                                 y=[som_state.codebook[r,c,1], som_state.codebook[r+1,c,1]],
                                                 mode='lines', line=dict(color='black', width=1), showlegend=False))
                    fig.add_trace(go.Scatter(x=[som_state.codebook[r,c,0]], y=[som_state.codebook[r,c,1]],
                                             mode='markers', marker=dict(color='black', size=8), showlegend=False))

        fig.update_layout(xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1]), height=450,
                          margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                          legend=dict(orientation="h", y=1.05))

        # BUILD ERROR FIGURE
        fig_err = go.Figure(go.Scatter(y=som_state.errors[-200:], mode='lines', line=dict(color='red')))
        fig_err.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10),
                               title="Quantization Error (Last 200 steps)")

        return (fig, fig_err, som_state.status_log,
                generate_code_display(som_state.phase, som_state.play_mode, som_state.topology),
                not som_state.play_mode, speed,
                build_umatrix_figure())


if __name__ == '__main__':
    _app = dash.Dash(__name__)
    _app.layout = layout
    register_callbacks(_app)
    _app.run(debug=True)
