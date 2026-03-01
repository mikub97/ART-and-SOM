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
X_iris = iris.data
np.random.shuffle(X_iris)
X_iris = X_iris[:15]

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
        self.phase_executed = False
        self.transition_to = None

        self.current_sample = np.zeros(self.n_inputs)
        self.I_sorted = []
        self.candidate_idx = 0
        self.winning_cat = None
        self.is_new = False
        self.status_log = "Continuous ART2A_E initialized.\n"

art2_state = ContinuousAppState()


def generate_code_display(phase, is_new):
    """Maps the current execution phase to specific lines of code for highlighting."""
    # Line indices in original_art2_code (0-based after split('\n')):
    #  0: class ART2A_E:
    #  3: def __init__(...)
    # 15: self.learning_rate = learning_rate
    # 17: def _find_best(self, X):
    # 19: self.F2[...] = 1 - np.sqrt(...)
    # 20: I = np.argsort(...)
    # 22: for i in I:
    # 24: if self.F2[i] >= self.rho:
    # 26: self.W[i] = self.learning_rate*X + ...
    # 27: return i
    # 29: def learn(self, X):
    # 31: # Standardize X to fit in [0,1] range
    # 34: X_ /= X_.max()
    # 41: # No match found...
    # 44: self.W[i] = X_
    # 46: return self.W[i], i
    lines = original_art2_code.split('\n')
    highlight_indices = []

    if phase == "INIT":
        highlight_indices = list(range(3, 16))       # __init__ body
    elif phase == "LOAD_INPUT":
        highlight_indices = [29, 30, 31, 32, 33, 34] # learn() + standardize
    elif phase == "BOTTOM_UP":
        highlight_indices = [17, 18, 19, 20]          # _find_best + distance
    elif phase == "VIGILANCE_CHECK":
        highlight_indices = [22, 23, 24]              # for loop + vigilance check
    elif phase == "UPDATE_WEIGHTS":
        if is_new:
            highlight_indices = [44, 45, 46]          # W[i] = X_, active += 1
        else:
            highlight_indices = [25, 26, 27]          # learn data (geometric update)
    elif phase == "CREATE_NEW":
        highlight_indices = [41, 42, 43]              # no match found, allocate slot

    components = []
    for i, line in enumerate(lines):
        style = {
            'paddingLeft': '10px', 'margin': '0', 'whiteSpace': 'pre',
            'fontFamily': 'monospace', 'fontSize': '13px'
        }
        if i in highlight_indices:
            style['backgroundColor'] = '#ffeeba'
            style['fontWeight'] = 'bold'
            style['borderLeft'] = '4px solid #ffc107'
        else:
            style['borderLeft'] = '4px solid transparent'
            style['color'] = '#555'
        components.append(html.Div(line, style=style))

    return components


# -----------------------------------------------------------------------------
# LAYOUT (importable)
# -----------------------------------------------------------------------------
layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'row', 'fontFamily': 'Arial, sans-serif',
           'padding': '20px', 'gap': '30px', 'height': '85vh', 'boxSizing': 'border-box'},
    children=[
        # LEFT PANEL: Original Code
        html.Div(style={'flex': '0 0 420px', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H3("Original ART2A_E Implementation Code",
                    style={'marginTop': '0', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),
            html.Div(id='art2-code-panel', style={
                'backgroundColor': '#f8f9fa', 'padding': '15px 0', 'border': '1px solid #ddd',
                'overflowY': 'auto', 'flex': '1', 'lineHeight': '1.6'
            })
        ]),

        # RIGHT PANEL: Interactive Dashboard
        html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H1("Continuous ART2A_E Sequential Dashboard",
                    style={'marginTop': '0', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),

            # Controls
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px',
                            'alignItems': 'center', 'paddingTop': '10px'}, children=[
                html.Div([
                    html.Label("Vigilance (Rho): ", style={'fontWeight': 'bold'}),
                    dcc.Slider(id='art2-rho-s', min=0.5, max=1.0, step=0.01, value=0.9,
                               marks={0.5: '0.5', 0.7: '0.7', 0.9: '0.9', 1.0: '1.0'})
                ], style={'flex': 1}),
                html.Div([
                    html.Label("Learning Rate (Eta): ", style={'fontWeight': 'bold'}),
                    dcc.Slider(id='art2-eta-s', min=0.05, max=0.5, step=0.05, value=0.2,
                               marks={0.1: '0.1', 0.3: '0.3', 0.5: '0.5'})
                ], style={'flex': 1}),
                html.Button("▶ Execute Step", id='art2-next-b', n_clicks=0,
                            style={'padding': '10px 20px', 'fontSize': '16px',
                                   'backgroundColor': '#007BFF', 'color': 'white',
                                   'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px',
                                   'minWidth': '180px'}),
                html.Button("Reset", id='art2-reset-b', n_clicks=0,
                            style={'padding': '10px 20px', 'fontSize': '16px',
                                   'backgroundColor': '#DC3545', 'color': 'white',
                                   'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px'}),
            ]),

            # Visualizations and Logs
            html.Div(style={'display': 'flex', 'gap': '20px', 'flex': '1'}, children=[
                dcc.Textarea(
                    id='art2-log-t', readOnly=True,
                    style={'width': '260px', 'height': '100%', 'fontFamily': 'monospace',
                           'fontSize': '12px', 'padding': '10px', 'boxSizing': 'border-box',
                           'resize': 'none', 'border': '1px solid #ccc'}
                ),
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
         Output('art2-code-panel', 'children'),
         Output('art2-next-b', 'children')],
        [Input('art2-next-b', 'n_clicks'),
         Input('art2-reset-b', 'n_clicks')],
        [State('art2-rho-s', 'value'),
         State('art2-eta-s', 'value')]
    )
    def step_continuous_art(n, r, rho, eta):
        tid = ctx.triggered_id

        if tid == 'art2-reset-b':
            art2_state.reset()

        elif tid == 'art2-next-b':
            if not art2_state.phase_executed:
                # ── EXECUTE: run computation, stay on current phase highlight ──
                if art2_state.phase == "INIT":
                    art2_state.status_log += "Initialization complete. Ready for inputs.\n"
                    art2_state.transition_to = "LOAD_INPUT"

                elif art2_state.phase == "LOAD_INPUT":
                    if art2_state.input_idx < len(X_iris):
                        raw = X_iris[art2_state.input_idx]
                        art2_state.current_sample = (raw - raw.min()) / (raw.max() - raw.min())
                        art2_state.F2 = np.zeros(art2_state.m_categories)
                        art2_state.winning_cat = None
                        art2_state.is_new = False
                        art2_state.status_log += f"\n--- Sample {art2_state.input_idx} ---\n"
                        art2_state.transition_to = "BOTTOM_UP"
                    else:
                        art2_state.status_log += "Finished all samples.\n"
                        art2_state.transition_to = None

                elif art2_state.phase == "BOTTOM_UP":
                    art2_state.F2 = 1 - np.sqrt(
                        ((art2_state.W - art2_state.current_sample) ** 2).sum(axis=1)
                    )
                    art2_state.I_sorted = np.argsort(
                        art2_state.F2[:art2_state.active].ravel()
                    )[::-1]
                    art2_state.candidate_idx = 0
                    art2_state.status_log += (
                        f"Similarities computed. Queue: {art2_state.I_sorted.tolist()}\n"
                    )
                    art2_state.transition_to = (
                        "VIGILANCE_CHECK" if art2_state.active > 0 else "CREATE_NEW"
                    )

                elif art2_state.phase == "VIGILANCE_CHECK":
                    if art2_state.candidate_idx < len(art2_state.I_sorted):
                        i = art2_state.I_sorted[art2_state.candidate_idx]
                        score = art2_state.F2[i]
                        art2_state.status_log += f"Check Cat {i}: Sim {score:.3f} (Req: {rho})\n"
                        if score >= rho:
                            art2_state.status_log += f"-> RESONANCE with Cat {i}.\n"
                            art2_state.winning_cat = i
                            art2_state.is_new = False
                            art2_state.transition_to = "UPDATE_WEIGHTS"
                        else:
                            art2_state.status_log += f"-> RESET Cat {i}.\n"
                            art2_state.candidate_idx += 1
                            if art2_state.candidate_idx >= len(art2_state.I_sorted):
                                art2_state.status_log += "All categories failed.\n"
                                art2_state.transition_to = "CREATE_NEW"
                            else:
                                art2_state.transition_to = None  # stay: check next candidate
                    else:
                        art2_state.transition_to = "CREATE_NEW"

                elif art2_state.phase == "CREATE_NEW":
                    if art2_state.active < art2_state.m_categories:
                        art2_state.winning_cat = art2_state.active
                        art2_state.is_new = True
                        art2_state.active += 1
                        art2_state.status_log += f"Created Cat {art2_state.winning_cat}.\n"
                        art2_state.transition_to = "UPDATE_WEIGHTS"
                    else:
                        art2_state.status_log += "Capacity full. Skipping input.\n"
                        art2_state.input_idx += 1
                        art2_state.transition_to = "LOAD_INPUT"

                elif art2_state.phase == "UPDATE_WEIGHTS":
                    i = art2_state.winning_cat
                    if art2_state.is_new:
                        art2_state.W[i] = art2_state.current_sample
                    else:
                        art2_state.W[i] = eta * art2_state.current_sample + (1 - eta) * art2_state.W[i]
                    art2_state.status_log += "Weights updated.\n"
                    art2_state.input_idx += 1
                    art2_state.transition_to = "LOAD_INPUT"

                art2_state.phase_executed = True

            else:
                # ── ADVANCE: move highlight to the next phase, no computation ──
                if art2_state.transition_to is not None:
                    art2_state.phase = art2_state.transition_to
                    art2_state.transition_to = None
                art2_state.phase_executed = False

        # ── Button label ──────────────────────────────────────────────────────
        btn_label = "→ Next Phase" if art2_state.phase_executed else "▶ Execute Step"

        # ── Determine active category for highlights ──────────────────────────
        active_cat = None
        if art2_state.phase in ["VIGILANCE_CHECK", "CREATE_NEW", "UPDATE_WEIGHTS"]:
            if art2_state.winning_cat is not None:
                active_cat = art2_state.winning_cat
            elif (art2_state.phase == "VIGILANCE_CHECK"
                  and art2_state.candidate_idx < len(art2_state.I_sorted)):
                active_cat = art2_state.I_sorted[art2_state.candidate_idx]

        # ── Generate Figure ───────────────────────────────────────────────────
        annotation_font = dict(size=10)
        n = art2_state.n_inputs
        m = art2_state.m_categories

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Normalized Input X",
                "Similarity (1 − Distance)",
                "Prototypes W Matrix",
                "Category Similarity vs Vigilance"
            ),
            vertical_spacing=0.26, horizontal_spacing=0.18
        )

        # Row 1, Col 1 — Normalized Input X (1 × n)
        fig.add_trace(go.Heatmap(
            z=art2_state.current_sample.reshape(1, -1),
            colorscale='Greys', zmin=0, zmax=1,
            text=art2_state.current_sample.reshape(1, -1),
            texttemplate="%{text:.2f}", textfont=annotation_font,
            showscale=False, xgap=1, ygap=1
        ), row=1, col=1)

        # Row 1, Col 2 — F2 Similarities (1 × m)
        fig.add_trace(go.Heatmap(
            z=art2_state.F2.reshape(1, -1),
            colorscale='Purples', zmin=0, zmax=1,
            text=art2_state.F2.reshape(1, -1),
            texttemplate="%{text:.2f}", textfont=dict(size=8),
            showscale=False, xgap=1, ygap=1
        ), row=1, col=2)

        # Row 2, Col 1 — Prototypes W (m × n)
        fig.add_trace(go.Heatmap(
            z=art2_state.W,
            colorscale='Blues', zmin=0, zmax=1,
            text=art2_state.W,
            texttemplate="%{text:.2f}", textfont=annotation_font,
            showscale=False, xgap=1, ygap=1
        ), row=2, col=1)

        # Row 2, Col 2 — Bar chart: similarity per category with vigilance line
        bar_colors = ['#d0d0d0'] * m
        for j in range(art2_state.active):
            bar_colors[j] = '#7b2d8b'
        if active_cat is not None:
            bar_colors[active_cat] = '#e63946'

        fig.add_trace(go.Bar(
            x=list(range(m)),
            y=art2_state.F2,
            marker_color=bar_colors,
            showlegend=False
        ), row=2, col=2)

        fig.add_hline(
            y=rho, line_dash="dash", line_color="orange", line_width=2,
            annotation_text=f"ρ={rho}", annotation_position="top right",
            row=2, col=2
        )

        # ── Red highlight boxes ───────────────────────────────────────────────
        if active_cat is not None:
            # Highlight column in F2 heatmap (row 1, col 2)
            fig.add_shape(
                type="rect",
                x0=active_cat - 0.5, y0=-0.5, x1=active_cat + 0.5, y1=0.5,
                line=dict(color="red", width=3), row=1, col=2
            )
            # Highlight row in W matrix (row 2, col 1)
            fig.add_shape(
                type="rect",
                x0=-0.5, y0=active_cat - 0.5, x1=n - 0.5, y1=active_cat + 0.5,
                line=dict(color="red", width=3), row=2, col=1
            )

        # ── Axis labels ───────────────────────────────────────────────────────
        fig.update_xaxes(title_text="Features (n)", row=1, col=1, tickfont=dict(size=9))
        fig.update_yaxes(showticklabels=False, row=1, col=1)

        fig.update_xaxes(title_text="Categories (m)", row=1, col=2,
                         tickfont=dict(size=9), dtick=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)

        fig.update_xaxes(title_text="Features (n)", row=2, col=1, tickfont=dict(size=9))
        fig.update_yaxes(title_text="Categories (m)", row=2, col=1,
                         tickfont=dict(size=9), dtick=1)

        fig.update_xaxes(title_text="Category", row=2, col=2,
                         tickfont=dict(size=9), dtick=1)
        fig.update_yaxes(title_text="Similarity", row=2, col=2,
                         range=[-0.05, 1.1], tickfont=dict(size=9))

        fig.update_annotations(font_size=11)  # smaller subplot titles prevent overlap
        fig.update_layout(margin=dict(l=30, r=40, t=55, b=30), autosize=True)

        return (
            fig,
            art2_state.status_log,
            generate_code_display(art2_state.phase, art2_state.is_new),
            btn_label
        )


if __name__ == '__main__':
    _app = dash.Dash(__name__)
    _app.layout = layout
    register_callbacks(_app)
    _app.run(debug=True)
