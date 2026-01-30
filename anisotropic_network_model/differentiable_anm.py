import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import urllib.request

PDB_ID = "1UBQ"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE}")


def parse_pdb_with_bfactors(pdb_file):
    coords, bfactors, residues = [], [], []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                chain = line[21]
                if atom_name == "CA" and chain == "A":
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    bfactors.append(float(line[60:66]))
                    residues.append(line[17:20].strip())
    return np.array(coords), np.array(bfactors), residues


def build_hessian_vectorized(coords, cutoff, gamma):
    n_atoms = coords.shape[0]
    device = coords.device
    dtype = coords.dtype
    
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)
    dist = torch.sqrt(dist_sq + 1e-8)
    
    # Much steeper sigmoid — closer to hard cutoff
    steepness = 20.0
    weights = torch.sigmoid(steepness * (cutoff - dist))
    weights = weights * (1 - torch.eye(n_atoms, device=device, dtype=dtype))
    
    outer = torch.einsum('ijk,ijl->ijkl', diff, diff)
    
    # Add numerical stability
    scale = -gamma * weights / (dist_sq + 1e-8)
    scale = torch.clamp(scale, min=-1e6, max=1e6)
    
    blocks = scale.unsqueeze(-1).unsqueeze(-1) * outer
    
    hessian = blocks.permute(0, 2, 1, 3).reshape(n_atoms * 3, n_atoms * 3)
    
    diag_blocks = -blocks.sum(dim=1)
    idx = torch.arange(n_atoms, device=device)
    for d1 in range(3):
        for d2 in range(3):
            hessian[idx * 3 + d1, idx * 3 + d2] = diag_blocks[:, d1, d2]
    
    return hessian


def compute_bfactors_from_anm(coords, cutoff, gamma, n_modes=20):
    n_atoms = coords.shape[0]
    
    hessian = build_hessian_vectorized(coords, cutoff, gamma)
    
    # Check for NaN before eigensolve
    if torch.isnan(hessian).any():
        return torch.full((n_atoms,), float('nan'), device=coords.device)
    
    hessian_cpu = hessian.to('cpu')
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian_cpu)
    eigenvalues = eigenvalues.to(coords.device)
    eigenvectors = eigenvectors.to(coords.device)
    
    valid_modes = slice(6, 6 + n_modes)
    evals = eigenvalues[valid_modes]
    
    # Clamp eigenvalues to avoid division issues
    evals = torch.clamp(evals, min=1e-8)
    
    evecs = eigenvectors[:, valid_modes].reshape(n_atoms, 3, n_modes)
    sq_disp = (evecs ** 2).sum(dim=1)
    weights = 1.0 / evals
    bfactors = (sq_disp * weights).sum(dim=1)
    
    return bfactors


def fit_anm_to_bfactors(coords_np, exp_bfactors, n_steps=200, lr=0.1):
    coords = torch.tensor(coords_np, dtype=torch.float32, device=DEVICE)
    exp_bf = torch.tensor(exp_bfactors, dtype=torch.float32, device=DEVICE)
    exp_bf_norm = (exp_bf - exp_bf.mean()) / exp_bf.std()
    
    # Start closer to known good values
    cutoff = torch.nn.Parameter(torch.tensor(8.0, dtype=torch.float32, device=DEVICE))
    gamma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
    
    optimizer = torch.optim.Adam([cutoff, gamma], lr=lr)
    
    history = {'cutoff': [], 'gamma': [], 'correlation': []}
    best_corr = -1
    best_params = (8.0, 1.0)
    
    print(f"{'Step':>6} {'Cutoff':>10} {'Gamma':>10} {'Corr':>10}")
    print("-" * 45)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        pred_bf = compute_bfactors_from_anm(coords, cutoff, gamma, n_modes=20)
        
        # Skip if NaN
        if torch.isnan(pred_bf).any():
            print(f"NaN at step {step}, stopping early")
            break
        
        pred_bf_norm = (pred_bf - pred_bf.mean()) / (pred_bf.std() + 1e-8)
        
        correlation = torch.corrcoef(torch.stack([pred_bf_norm, exp_bf_norm]))[0, 1]
        
        loss = -correlation
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([cutoff, gamma], max_norm=1.0)
        
        optimizer.step()
        
        # Strict parameter bounds
        with torch.no_grad():
            cutoff.clamp_(5.0, 15.0)
            gamma.clamp_(0.1, 5.0)
        
        corr_val = correlation.item()
        history['cutoff'].append(cutoff.item())
        history['gamma'].append(gamma.item())
        history['correlation'].append(corr_val)
        
        if corr_val > best_corr:
            best_corr = corr_val
            best_params = (cutoff.item(), gamma.item())
        
        if step % 20 == 0:
            print(f"{step:>6} {cutoff.item():>10.2f} {gamma.item():>10.3f} {corr_val:>10.3f}")
    
    print("-" * 45)
    print(f"Best: cutoff={best_params[0]:.2f}Å, gamma={best_params[1]:.3f}, corr={best_corr:.3f}")
    
    # Compute final prediction with best params
    with torch.no_grad():
        final_pred = compute_bfactors_from_anm(
            coords, 
            torch.tensor(best_params[0], device=DEVICE),
            torch.tensor(best_params[1], device=DEVICE),
            n_modes=20
        )
    
    return best_params[0], best_params[1], final_pred.cpu().numpy(), history


def plot_results(exp_bfactors, pred_bfactors, history, filename="ubq_bfactor_fit.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    exp_norm = (exp_bfactors - exp_bfactors.mean()) / exp_bfactors.std()
    pred_norm = (pred_bfactors - pred_bfactors.mean()) / pred_bfactors.std()
    
    ax1 = axes[0, 0]
    ax1.plot(exp_norm, 'b-', label='Experimental', linewidth=2)
    ax1.plot(pred_norm, 'r--', label='ANM Predicted', linewidth=2)
    ax1.set_xlabel('Residue Index')
    ax1.set_ylabel('Normalized B-factor')
    ax1.set_title('B-factors Along Sequence')
    ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.scatter(exp_norm, pred_norm, alpha=0.6, s=40)
    ax2.plot([-2.5, 3.5], [-2.5, 3.5], 'k--', alpha=0.5)
    corr = np.corrcoef(exp_norm, pred_norm)[0, 1]
    ax2.set_xlabel('Experimental')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Correlation: {corr:.3f}')
    
    ax3 = axes[1, 0]
    ax3.plot(history['correlation'], 'g-')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Training Progress')
    ax3.axhline(y=0.757, color='gray', linestyle='--', label='Grid search best (8Å)')
    ax3.legend()
    
    ax4 = axes[1, 1]
    ax4.plot(history['cutoff'], label='Cutoff (Å)')
    ax4.plot(history['gamma'], label='Gamma')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Value')
    ax4.set_title('Parameter Evolution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    pdb_file = f"{PDB_ID}.pdb"
    if not os.path.exists(pdb_file):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{PDB_ID}.pdb", pdb_file)
    
    coords, exp_bfactors, residues = parse_pdb_with_bfactors(pdb_file)
    print(f"Loaded {PDB_ID}: {len(coords)} residues")
    
    cutoff, gamma, pred_bfactors, history = fit_anm_to_bfactors(coords, exp_bfactors)
    
    plot_results(exp_bfactors, pred_bfactors, history, "ubq_bfactor_fit.png")
    
    print(f"\nGrid search: r=0.757 at 8.0Å, gamma=1.0")
    print(f"Gradient descent: r={max(history['correlation']):.3f} at {cutoff:.1f}Å, gamma={gamma:.2f}")