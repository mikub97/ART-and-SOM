import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.datasets import load_iris

# -----------------------------------------------------------------------------
# ORIGINAL ART2A_E IMPLEMENTATION
# -----------------------------------------------------------------------------
original_art2_code = """class ART2A_E:
    ''' ART class for continuous data '''

    def __init__(self, n=5, m=10, rho=.5, learning_rate=0.2):
        # Comparison layer
        self.F1 = np.ones(n)
        # Recognition layer
        self.F2 = np.ones(m)
        # Prototypes
        self.W = np.ones((m,n))
        # Vigilance
        self.rho = rho
        # Number of active units in F2
        self.active = 0
        # Learning rate
        self.learning_rate = learning_rate

    def _find_best(self, X):
        # Compute F2 outputs as distances from prototype and sort them (I)
        self.F2[...] = 1 - np.sqrt(((self.W - X)**2).sum(axis=1))
        I = np.argsort(self.F2[:self.active].ravel())[::-1]

        for i in I:
            # Check if nearest memory is above the vigilance level
            if self.F2[i] >= self.rho:
                # Learn data
                self.W[i] = self.learning_rate*X + (1-self.learning_rate)*self.W[i]
                return i

    def learn(self, X):
        ''' Learn X '''
        # Standardize X to fit in [0,1] range
        X_ = X - X.min()
        if X_.max() > 0:
            X_ /= X_.max()

        # Find best prototype
        i = self._find_best(X_)
        if i is not None:
            return self.W[i], i

        # No match found, increase the number of active units
        if self.active < self.F2.size:
            i = self.active
            self.W[i] = X_
            self.active += 1
            return self.W[i], i

        return None, None"""

# Load Iris Data for simulation
iris = load_iris()

X_iris = iris.data # Take first 15 samples for visualization
np.random.shuffle(X_iris)
X_iris=X_iris[:15]

# -----------------------------------------------------------------------------
# APP STATE MANAGER
# -----------------------------------------------------------------------------
class ContinuousAppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.input_idx = 0
        self.n_inputs = 4
        self.m_categories = 6

        self.W = np.ones((self.m_categories, self.n_inputs))
        self.F2 = np.zeros(self.m_categories)
        self.active = 0

        self.phase = "INIT"
        self.current_sample = np.zeros(self.n_inputs)
        self.I_sorted = []
        self.candidate_idx = 0
        self.winning_cat = None
        self.is_new = False
        self.status_log = "Continuous ART2A_E initialized.\n"

art2_state = ContinuousAppState()

def highlight_art2_code(phase, is_new):
    lines = original_art2_code.split('\n')
    highlights = []

    if phase == "INIT": highlights = list(range(3, 14))
    elif phase == "LOAD_INPUT": highlights = [31, 34, 35, 36] # Standardize
    elif phase == "BOTTOM_UP": highlights = [16, 17, 18] # Distance computation
    elif phase == "VIGILANCE_CHECK": highlights = [20, 21] # Match check
    elif phase == "CREATE_NEW": highlights = [44, 45, 46, 47, 48] # New node
    elif phase == "UPDATE_WEIGHTS":
        if is_new: highlights = [46, 47]
        else: highlights = [22, 23, 24] # Geometric update

    return [html.Div(line, style={
        'paddingLeft': '10px', 'whiteSpace': 'pre', 'fontFamily': 'monospace', 'fontSize': '12px',
        'backgroundColor': '#ffeeba' if i in highlights else 'transparent',
        'fontWeight': 'bold' if i in highlights else 'normal',
        'borderLeft': '4px solid #ffc107' if i in highlights else '4px solid transparent'
    }) for i, line in enumerate(lines)]

# -----------------------------------------------------------------------------
# LAYOUT (importable)
# -----------------------------------------------------------------------------
layout = html.Div(
    style={'display': 'flex', 'padding': '20px', 'gap': '20px', 'height': '85vh'},
    children=[
        # Code Panel
        html.Div(style={'flex': '0 0 450px', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H3("Original ART2A_E Code"),
            html.Div(id='art2-code-panel', style={
                'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'flex': '1', 'overflowY': 'auto'
            })
        ]),

        # Visualization Panel
        html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H2("Continuous ART2A_E Sequential Dashboard"),

            html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '15px'}, children=[
                html.Div([html.B("Vigilance (Rho):"),
                          dcc.Slider(id='art2-rho-s', min=0.5, max=1.0, step=0.01, value=0.9,
                                     marks={0.5: '0.5', 1: '1.0'})], style={'flex': 1}),
                html.Div([html.B("Learning Rate (Eta):"),
                          dcc.Slider(id='art2-eta-s', min=0.05, max=0.5, step=0.05, value=0.2,
                                     marks={0.1: '0.1', 0.5: '0.5'})], style={'flex': 1}),
                html.Button("Next Step", id='art2-next-b', n_clicks=0,
                            style={'padding': '10px 20px', 'backgroundColor': '#007BFF',
                                   'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
                html.Button("Reset", id='art2-reset-b', n_clicks=0,
                            style={'padding': '10px 20px', 'backgroundColor': '#DC3545',
                                   'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
            ]),

            html.Div(style={'display': 'flex', 'gap': '20px', 'flex': '1'}, children=[
                dcc.Textarea(id='art2-log-t', readOnly=True,
                             style={'width': '250px', 'height': '100%', 'fontSize': '11px'}),
                dcc.Graph(id='art2-graph-v', style={'flex': '1', 'height': '100%'})
            ])
        ])
    ]
)

# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------
def register_callbacks(app):
    @app.callback(
        [Output('art2-graph-v', 'figure'),
         Output('art2-log-t', 'value'),
         Output('art2-code-panel', 'children')],
        [Input('art2-next-b', 'n_clicks'),
         Input('art2-reset-b', 'n_clicks')],
        [State('art2-rho-s', 'value'),
         State('art2-eta-s', 'value')]
    )
    def step_continuous_art(n, r, rho, eta):
        tid = ctx.triggered_id
        if tid == 'art2-reset-b': art2_state.reset()
        elif tid == 'art2-next-b':
            if art2_state.phase == "INIT":
                art2_state.phase = "LOAD_INPUT"
                art2_state.status_log += "Initialization complete.\n"
            elif art2_state.phase == "LOAD_INPUT":
                if art2_state.input_idx < len(X_iris):
                    raw = X_iris[art2_state.input_idx]
                    art2_state.current_sample = (raw - raw.min()) / (raw.max() - raw.min()) # Standardize
                    art2_state.status_log += f"\n--- Sample {art2_state.input_idx} ---\n"
                    art2_state.phase = "BOTTOM_UP"
                else: art2_state.status_log += "Finished sequence.\n"
            elif art2_state.phase == "BOTTOM_UP":
                art2_state.F2 = 1 - np.sqrt(((art2_state.W - art2_state.current_sample)**2).sum(axis=1)) # Euclidean distance
                art2_state.I_sorted = np.argsort(art2_state.F2[:art2_state.active].ravel())[::-1]
                art2_state.candidate_idx = 0
                art2_state.phase = "VIGILANCE_CHECK" if art2_state.active > 0 else "CREATE_NEW"
            elif art2_state.phase == "VIGILANCE_CHECK":
                if art2_state.candidate_idx < len(art2_state.I_sorted):
                    i = art2_state.I_sorted[art2_state.candidate_idx]
                    score = art2_state.F2[i]
                    art2_state.status_log += f"Check Cat {i}: Similarity {score:.3f} (Req: {rho})\n"
                    if score >= rho:
                        art2_state.status_log += f"-> Resonance with {i}.\n"
                        art2_state.winning_cat, art2_state.is_new, art2_state.phase = i, False, "UPDATE_WEIGHTS"
                    else: art2_state.candidate_idx += 1
                else: art2_state.phase = "CREATE_NEW"
            elif art2_state.phase == "CREATE_NEW":
                if art2_state.active < art2_state.m_categories:
                    art2_state.winning_cat, art2_state.active, art2_state.is_new, art2_state.phase = art2_state.active, art2_state.active + 1, True, "UPDATE_WEIGHTS"
                    art2_state.status_log += f"Created Cat {art2_state.winning_cat}.\n"
                else: art2_state.phase, art2_state.input_idx = "LOAD_INPUT", art2_state.input_idx + 1
            elif art2_state.phase == "UPDATE_WEIGHTS":
                i = art2_state.winning_cat
                if art2_state.is_new: art2_state.W[i] = art2_state.current_sample
                else: art2_state.W[i] = eta * art2_state.current_sample + (1 - eta) * art2_state.W[i] # Geometric update
                art2_state.status_log += "Weights updated.\n"
                art2_state.phase, art2_state.input_idx = "LOAD_INPUT", art2_state.input_idx + 1

        # Figure generation
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Normalized Input X", "Similarity (1-Distance)", "Prototypes (W) Matrix", "Category Activation"))
        fnt = dict(size=10)

        fig.add_trace(go.Heatmap(z=art2_state.current_sample.reshape(1,-1), colorscale='Greys', texttemplate="%{z:.2f}", textfont=fnt, showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=art2_state.F2.reshape(1,-1), colorscale='Purples', texttemplate="%{z:.2f}", textfont=fnt, showscale=False), 1, 2)
        fig.add_trace(go.Heatmap(z=art2_state.W, colorscale='Blues', texttemplate="%{z:.2f}", textfont=fnt, showscale=False, xgap=1, ygap=1), 2, 1)

        # Highlight winning row
        if art2_state.phase == "UPDATE_WEIGHTS" or art2_state.phase == "VIGILANCE_CHECK":
            target = art2_state.winning_cat if art2_state.winning_cat is not None else (art2_state.I_sorted[art2_state.candidate_idx] if art2_state.candidate_idx < len(art2_state.I_sorted) else None)
            if target is not None:
                fig.add_shape(type="rect", x0=-0.5, y0=target-0.5, x1=3.5, y1=target+0.5, line=dict(color="red", width=3), row=2, col=1)

        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)

        return fig, art2_state.status_log, highlight_art2_code(art2_state.phase, art2_state.is_new)


if __name__ == '__main__':
    _app = dash.Dash(__name__)
    _app.layout = layout
    register_callbacks(_app)
    _app.run(debug=True)
