import numpy as np
import urllib.request
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

#set parameters
PDB_ID = "1BR1"       # Myosin Motor Domain
CUTOFF = 15.0         #interaction distance cutoffs
GAMMA = 1.0           #spring constant value
N_MODES = 1           #mode to visualize 

def parse_pdb_calpha(pdb_file):
    #parse pdb file to extract alpha carbon coordinates and return an array of shape N, 3. 
    coords = []
    residues = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                
                #get chain A's alpha carbons
                if atom_name == "CA" and chain == "A":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    residues.append(res_name)
                    
    return np.array(coords), residues

def build_hessian(coords, cutoff):
    #construct 3Nx3N hessian
    n_atoms = len(coords)
    n_dim = 3 * n_atoms
    hessian = np.zeros((n_dim, n_dim))
    
    #pairwise calculations for simplicity. this is O(N^2). Next step could be to try to vectorize this 
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            
            #i,j vector
            d_vec = coords[j] - coords[i]
            dist_sq = np.dot(d_vec, d_vec)
            dist = np.sqrt(dist_sq)
            
            if dist < cutoff:
                #Super-Element Formula, H_ij = -gamma * (outer_product(d_vec, d_vec)) / dist^2
                
                #Normalize force vector? 
                #Standard ANM: second derivative of harmonic potential, simplified often to purely geometric projection
                
                element = -GAMMA * np.outer(d_vec, d_vec) / dist_sq
                
                # Fill off-diagonal blocks (H_ij and H_ji)
                i_start, i_end = i*3, i*3+3
                j_start, j_end = j*3, j*3+3
                
                hessian[i_start:i_end, j_start:j_end] = element
                hessian[j_start:j_end, i_start:i_end] = element
                
                #accumulate diagonal blocks (H_ii and H_jj)
                #we know that the force on self is the opposite of force on neighbor
                hessian[i_start:i_end, i_start:i_end] -= element
                hessian[j_start:j_end, j_start:j_end] -= element
                
    return hessian


def save_mode_gif(coords, mode_vector, filename="anm_mode.gif", n_frames=60):
    """
    Renders the mode oscillation as a GIF with visible motion.
    """
    n_atoms = len(coords)
    deformation = mode_vector.reshape((n_atoms, 3))
    
    #scale eigenvector relative to protein size so motion is visible
    deformation_normalized = deformation / np.linalg.norm(deformation)
    protein_radius = np.ptp(coords, axis=0).max() / 2
    amplitude = protein_radius * 0.15  # 15% of protein size
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.plasma(np.linspace(0, 1, n_atoms))
    
    def update(frame):
        ax.clear()
        
        theta = 2 * np.pi * frame / n_frames
        factor = amplitude * np.sin(theta)
        current = coords + deformation_normalized * factor
        
        ax.plot(current[:, 0], current[:, 1], current[:, 2], 
                'k-', alpha=0.4, linewidth=1)
        ax.scatter(current[:, 0], current[:, 1], current[:, 2],
                   c=colors, s=15, depthshade=True)
        
        ax.view_init(elev=20, azim=45)
        
        center = coords.mean(axis=0)
        max_range = protein_radius + amplitude + 5
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'ANM Mode {N_MODES}')
        
        return []
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    anim.save(filename, writer=PillowWriter(fps=20))
    plt.close()
    print(f"Saved to {filename}")

#get data
pdb_filename = f"{PDB_ID}.pdb"
if not os.path.exists(pdb_filename):
    print(f"Downloading {PDB_ID}...")
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    urllib.request.urlretrieve(url, pdb_filename)

#parse data
coords, res = parse_pdb_calpha(pdb_filename)
n_atoms = len(coords)
print(f"Parsed {n_atoms} residues.")

#build hessian 
hessian = build_hessian(coords, CUTOFF)

#solve eigenproblem 
print("  > Diagonalizing Hessian (this involves the math engine)...")
eigenvalues, eigenvectors = np.linalg.eigh(hessian)

#extract the Slowest Non-Zero Mode
#i don't really care about the first 6 modes. objects in 3d space usually have 6 ways to move without internal deformation
#slide in x, y, z and rotation in x, y, z
#mode 7 (index 6) is the first real deformation.
target_mode_index = 6 + N_MODES - 1
mode_freq = eigenvalues[target_mode_index]
mode_vec = eigenvectors[:, target_mode_index]

print(f"Mode {N_MODES} (Index {target_mode_index})")
print(f"Eigenvalue (Stiffness): {mode_freq:.6f}")
save_mode_gif(coords, mode_vec, "myosin_mode1.gif")
