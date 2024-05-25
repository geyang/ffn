from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmx import doc

with doc:
    N = 500
    A = np.random.uniform(-1, 1, (N, N))
    eigenvalues = np.linalg.eigvalsh(A)

with doc:
    doc.print(max(eigenvalues), min(eigenvalues))

with doc:
    plt.hist(eigenvalues, bins=70)
    doc.savefig(f"{Path(__file__).stem}/wigner_circular_law.png")
