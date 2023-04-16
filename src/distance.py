
import similaritymeasures
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff(u, v):
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

def calculate_distance(inds, vehicles_):
    row = np.zeros(len(vehicles_))
    for u, v, val in inds:
        if val > 0:
            row[v] = np.inf
        else:
            row[v] = hausdorff(vehicles_[u], vehicles_[v])
    return u, row