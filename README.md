# ART & SOM— Interactive Dashboard

An interactive step-by-step visualiser for three classic unsupervised learning algorithms: **ART1**, **ART2A-E**, and **Self-Organising Maps (SOM)**. Each algorithm is presented as its own sub-dashboard, all combined into a single tabbed Dash application.

## Project Structure

```
ART/
├── dashboard.py      # Combined 3-tab Dash app (entry point)
├── ART1.py           # ART1 algorithm, layout, and callbacks
├── ART2A_E.py        # ART2A-E algorithm, layout, and callbacks
├── SOM.py            # SOM algorithm, layout, and callbacks
```

Each of `ART1.py`, `ART2A_E.py`, and `SOM.py` is **independently runnable** as a standalone dashboard, and also **importable** by `dashboard.py` via `layout` and `register_callbacks(app)`.

---

## Usage

### Combined dashboard (all three tabs)
```bash
python dashboard.py
```

### Individual dashboards
```bash
python ART1.py
python ART2A_E.py
python SOM.py
```

Then open `http://127.0.0.1:8050` in a browser.

---

## Controls

### ART1 & ART2A-E
| Control | Action |
|---------|--------|
| **▶ Execute Step** | Run the current highlighted phase and update the matrices |
| **→ Next Phase** | Advance the code highlight to the next phase (no computation) |
| **Reset Network** | Reinitialise all weights and restart |
| **Vigilance (ρ)** | Higher = more categories created (stricter matching) |

> The two-button cycle lets you see the **computation result in the matrices while the relevant code is still highlighted**, before advancing to the next step.

### SOM
| Control | Action |
|---------|--------|
| **Next Step** | Advance one algorithmic phase |
| **Play Sequence** | Toggle automatic playback |
| **Reset Network** | Reinitialise the codebook |
| **Neuron Count** | Number of neurons (total for 1D; nearest square for 2D) |
| **Sigma** | Neighbourhood radius |
| **Speed** | Milliseconds per step in play mode |

---

## Dependencies

```
dash
plotly
numpy
scikit-learn
```

Install with:
```bash
pip install dash plotly numpy scikit-learn
```
