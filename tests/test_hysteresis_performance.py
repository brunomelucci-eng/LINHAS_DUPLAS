import numpy as np
import time
from src.postprocessing.mask_cleaning import _hysteresis_threshold

def test_hysteresis_correctness_and_perf():
    # Create a small probability map
    prob = np.zeros((10, 10), dtype=np.float32)
    # A single connected seed region
    prob[2:5, 2:5] = 0.20 # candidates
    prob[3, 3] = 0.35      # strong seed
    
    # An isolated candidate region with no seed
    prob[7:9, 7:9] = 0.20
    
    # Apply hysteresis (low=0.15, high=0.30)
    result = _hysteresis_threshold(prob, low=0.15, high=0.30)
    
    # The seed region should be kept, the isolated region should be discarded
    assert result[3, 3] == 1
    assert result[2, 2] == 1
    assert result[7, 7] == 0
    assert result[8, 8] == 0
    
    # Verify performance threshold
    sz = 1000
    prob_large = np.random.rand(sz, sz).astype(np.float32)
    t0 = time.perf_counter()
    _ = _hysteresis_threshold(prob_large, low=0.12, high=0.28)
    elapsed = time.perf_counter() - t0
    
    # Large hysteresis thresholding on 1M pixels should take less than 0.5 seconds
    assert elapsed < 0.5
