
import similaritymeasures
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff(u, v):
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

def calculate_distance(i, u, vehicles_):
    row = np.zeros(len(vehicles_))
    for j in range(i + 1, len(vehicles_)):
        # row[j] = similaritymeasures.frechet_dist(u, vehicles_[j])
        row[j] = hausdorff(u, vehicles_[j])
    return i, row