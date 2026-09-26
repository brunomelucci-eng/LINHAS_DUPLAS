import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def prune_spurs(
    branches: List[List[Tuple[int, int]]],
    junctions: List[Tuple[int, int]],
    endpoints: List[Tuple[int, int]],
    gsd: float,
    config: dict
) -> List[List[Tuple[int, int]]]:
    """
    Remove short dead-end branches (spurs) based on physical length in meters.
    A branch is a spur if it has a dead-end endpoint (degree 1) at either start or end,
    and its total length is below min_component_length_m.
    """
    post_cfg = config.get('postprocessing', {})
    # Use dedicated spur parameter; fall back to legacy with deprecation warning
    if 'min_spur_length_m' in post_cfg:
        min_len_m = post_cfg['min_spur_length_m']
    elif 'min_component_length_m' in post_cfg:
        min_len_m = post_cfg['min_component_length_m']
        import logging as _log
        _log.getLogger(__name__).warning(
            "DEPRECATED: using 'min_component_length_m' for spur pruning. "
            "Switch to 'min_spur_length_m' in postprocessing config."
        )
    else:
        min_len_m = 0.50  # sensible default: 50 cm
    
    j_set = set(junctions)
    e_set = set(endpoints)
    
    cleaned_branches = []
    removed_count = 0
    
    for branch in branches:
        if len(branch) < 2:
            continue
            
        coords = np.array(branch)
        diffs = np.diff(coords, axis=0)
        # Scale to meters using GSD
        dists = np.sqrt(np.sum(diffs**2, axis=1)) * gsd
        total_len_m = dists.sum()
        
        start_node = branch[0]
        end_node = branch[-1]
        
        # Check if the ends correspond to degree-1 endpoints in the graph
        is_dead_end_start = start_node in e_set
        is_dead_end_end = end_node in e_set
        
        is_spur = (is_dead_end_start or is_dead_end_end)
        
        # Remove if it's a short dead-end branch or a tiny isolated segment
        if is_spur and total_len_m < min_len_m:
            removed_count += 1
        else:
            cleaned_branches.append(branch)
            
    logger.debug(f"Spur removal: pruned {removed_count} short branches.")
    return cleaned_branches
