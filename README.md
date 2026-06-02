# Fuzzy Set Markov Engine for ETFs

Applies fuzzy C‑means clustering to ETF returns to identify multiple fuzzy regimes. The per‑ETF score is the membership degree in the highest‑variance (extreme) regime on the last day of the window. This provides a soft, probabilistic measure of regime alignment.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Fuzzy C‑Means clustering (Bezdek, 1981)
- Identifies extreme regime by cluster variance
- Score = membership degree in extreme regime
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-fuzzy-set-markov-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High membership → ETF is deeply in the extreme (high‑volatility) regime → may signal upcoming large moves.
- Low membership → ETF is in a calm regime.

## Requirements

See `requirements.txt`.
