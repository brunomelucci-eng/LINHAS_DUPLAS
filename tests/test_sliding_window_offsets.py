from src.inference.sliding_window import generate_offsets

def test_generate_offsets_no_duplication():
    # Scenario listed in V3 audit: H=896, tile=512, step=384
    # Step = 512 - 128 = 384
    # Standard range would yield [0, 384, 384] if last offset is blindly replaced.
    # The new generate_offsets should output [0, 384] with no duplication!
    offsets = generate_offsets(size=896, tile_size=512, step=384)
    
    assert offsets == [0, 384]
    
    # Check simple small cases
    assert generate_offsets(size=400, tile_size=512, step=100) == [0]
    
    # Check H=1024, tile=512, step=256
    # 0, 256, 512. last is 1024 - 512 = 512.
    assert generate_offsets(size=1024, tile_size=512, step=256) == [0, 256, 512]
    
    # Check random dimensions from audit (e.g. 600, 800)
    offsets_600 = generate_offsets(size=600, tile_size=512, step=100)
    assert offsets_600 == [0, 88] # last offset is 600 - 512 = 88. 0 + 100 > 88, so only [0, 88]
    assert len(offsets_600) == len(set(offsets_600))
    assert min(offsets_600) >= 0
    
    # Ensure sorted order and no duplicates
    offsets_large = generate_offsets(size=14934, tile_size=512, step=384)
    assert len(offsets_large) == len(set(offsets_large))
    assert offsets_large == sorted(offsets_large)
    assert offsets_large[-1] == 14934 - 512
