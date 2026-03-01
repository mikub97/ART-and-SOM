# Neural Learning Algorithms — Interactive Dashboard

An interactive step-by-step visualiser for three classic unsupervised learning algorithms: **ART1**, **ART2A-E**, and **Self-Organising Maps (SOM)**. Each algorithm is presented as its own sub-dashboard, all combined into a single tabbed Dash application.

---

## Algorithms

### ART1 — Binary Adaptive Resonance Theory
Classifies binary (0/1) input patterns into categories using a two-layer competitive network.

**Key components**
| Matrix | Role |
|--------|------|
| `Wf` (feed-forward) | Bottom-up weights from input to recognition layer F2 |
| `Wb` (feed-back) | Top-down prototype weights from F2 back to comparison layer F1 |
| `F2` | Recognition layer activations |

**Algorithm phases**
1. **LOAD INPUT** — present a binary pattern X to F1
2. **BOTTOM-UP** — compute `F2 = Wf · X`; rank active categories by activation
3. **VIGILANCE CHECK** — for each candidate category *i*, compute the match score `d = (Wb[:,i] · X) / sum(X)` and compare against the vigilance threshold ρ
   - **Resonance (d ≥ ρ)** — update weights: `Wb[:,i] *= X`, then re-normalise `Wf[i,:] = Wb[:,i] / (0.5 + sum(Wb[:,i]))`
   - **Reset (d < ρ)** — inhibit category *i* and check the next candidate
4. **CREATE NEW** — if all active categories fail, allocate the next uncommitted node and update its weights immediately (same step)

> **Why the Wf normalisation formula?**
> `Wf[i,:] = Wb[:,i] / (0.5 + sum(Wb[:,i]))` solves the *superset bias* problem. A category with many active bits would otherwise always beat a smaller, more specific one. Dividing by the prototype's bit-count ensures that a perfect match to a specific pattern wins over a partial match to a broader template (Weber Law / Subset Rule).

---

### ART2A-E — Continuous Adaptive Resonance Theory
Extends ART to continuous-valued inputs (e.g. Iris dataset features). Prototypes are represented as real-valued vectors updated via a convex combination (geometric learning).

**Key components**
| Symbol | Role |
|--------|------|
| `W` | Prototype matrix (m categories × n features) |
| `F2` | Similarity scores `1 − ‖W − X‖₂` |
| ρ (rho) | Vigilance threshold |
| η (eta) | Learning rate for prototype update |

**Algorithm phases** mirror ART1 (load → bottom-up similarity → vigilance → create/update), with the weight update replaced by:
```
W[i] = η · X + (1 − η) · W[i]
```

---

### SOM — Self-Organising Map
Topology-preserving competitive learning on the 2D Iris feature space.

**Topologies**
- **1D Chain** — neurons arranged in a line; neighbourhood distance is `|i − winner|`
- **2D Grid** — neurons on a rectangular grid; neighbourhood distance is Euclidean in grid space

**Algorithm phases**
1. **LOAD SAMPLE** — pick a data point X
2. **FIND BMU** — Best Matching Unit = `argmin ‖codebook − X‖²`
3. **CALC NEIGHBOURHOOD** — Gaussian bubble `G = exp(−dist² / 2σ²)` centred on the BMU
4. **UPDATE WEIGHTS** — `codebook += lr · G · (X − codebook)`

**Play mode** runs the full loop automatically at an adjustable speed; the quantisation error plot tracks learning progress.

---

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
