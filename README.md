# lab-codes

> **My semester's worth of Neural & Deep Learning lab notebooks. Kept around so I don't have to rewrite the perceptron from scratch every time someone asks "wait how does Hebbian learning work again?"**

Just a dump of the labs I worked through for the NDL (Neural & Deep Learning) module. Nothing here is novel — it's the standard from-scratch progression every ML student does at some point: logic gates with a perceptron, XOR with a hidden layer, Hebbian PCA, MNIST with a feedforward net. The point was building intuition for *why* each layer of abstraction exists before reaching for `torch.nn.Sequential`.

## What's in here

```
ndl/
├── LogicGates.ipynb       Single perceptron solving AND/OR/NAND. The "why XOR is hard" setup
├── RevisionXOR.ipynb      XOR with one hidden layer — the punchline to the perceptron limitation
├── hebbian_pca.ipynb      Hebbian learning rule deriving principal components from raw covariance
├── Lab 7.ipynb            MNIST classification, the rite-of-passage notebook
├── lab-1-6.py             Earlier labs collapsed into one file once I got tired of notebook-per-lab
└── MNIST/                 The dataset, vendored in so the notebooks run offline
```

## Why this is public

It's not a portfolio piece — it's a study log. If you're a junior taking the same module and you're stuck on, say, why your Hebbian update keeps blowing up (hint: normalize), this is one more reference point alongside the textbook. Nothing here is the "right" implementation; it's the version that finally clicked for me.

## Running

Each notebook is self-contained. Spin up a venv, install torch + numpy + matplotlib, open Jupyter, run cells top-to-bottom. The MNIST lab expects `ndl/MNIST/raw/` to be sitting next to the notebook, which it is.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib jupyter
jupyter lab ndl/
```
