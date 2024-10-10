import numpy as np

def generate_edge(grid):
    grid = np.unique(grid)
    griddiff_half = np.diff(grid) / 2
    edgelen = np.concatenate((griddiff_half[0:1], griddiff_half)) + np.concatenate((griddiff_half, griddiff_half[-1:]))
    edge = np.full(len(grid)+1, grid[0]-griddiff_half[0])
    edge[1:] = edge[1:] + np.round(np.cumsum(edgelen), 2)
    edge = np.round(edge, 2)
    return edge