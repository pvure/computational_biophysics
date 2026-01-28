# Computational Biophysics

Implementations of core biophysics algorithms built from scratch in Python.

### 1. Anisotropic Network Model (ANM)
* **File:** `/anisotropic_network_model/anm.py`
* **Concept:** Uses a harmonic spring network model to predict the functional motions of a protein structure (Normal Mode Analysis).
* **Method:** Constructs a Hessian matrix from C-alpha coordinates and performs eigen-decomposition to identify low-frequency, large-scale fluctuations (domain movements) without expensive MD simulation.
