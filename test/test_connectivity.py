import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature.connectivity import pearson_connectivity, plv_connectivity, threshold_matrix

# 模拟一个 segment
np.random.seed(0)
fake_segment = np.random.randn(32, 256)

# 1️⃣ Pearson
pearson_adj = pearson_connectivity(fake_segment)
print("Pearson shape:", pearson_adj.shape)
print("Pearson first row:", pearson_adj[0])

# 2️⃣ PLV
plv_adj = plv_connectivity(fake_segment)
print("PLV shape:", plv_adj.shape)
print("PLV first row:", plv_adj[0])

# 3️⃣ Threshold
plv_thresh = threshold_matrix(plv_adj, thresh=0.5)
print("Thresholded PLV first row:", plv_thresh[0])
