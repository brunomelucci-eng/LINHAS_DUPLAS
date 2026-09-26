import pytest
import numpy as np
import networkx as nx
from src.postprocessing.skeleton_graph import skeleton_to_graph, extract_branches
from src.postprocessing.spur_removal import prune_spurs

def test_skeleton_to_graph_simple_line():
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    skeleton[2, 2:7] = 1  # 5 pixels horizontal line
    
    G = skeleton_to_graph(skeleton)
    assert len(G.nodes) == 5
    assert len(G.edges) == 4
    
    branches, junctions, endpoints = extract_branches(G)
    assert len(branches) == 1
    assert len(junctions) == 0
    assert len(endpoints) == 2
    assert len(branches[0]) == 5

def test_prune_spurs():
    # Construct lists representing branches, endpoints, and junctions
    # 1 long branch of 10 pixels, 1 short dead-end spur branch of 2 pixels
    branches = [
        [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)],
        [(1, 5), (2, 5), (3, 5)] # spur starting at junction (1,5)
    ]
    junctions = [(1, 5)]
    endpoints = [(1, 1), (1, 10), (3, 5)]
    
    config = {
        'postprocessing': {
            'min_component_length_m': 4.0
        }
    }
    
    # GSD = 1.0 -> spur is 2m long, main line is 9m long
    cleaned = prune_spurs(branches, junctions, endpoints, gsd=1.0, config=config)
    
    # The 2m spur branch should be pruned, leaving only the main branch
    assert len(cleaned) == 1
    assert len(cleaned[0]) == 10
