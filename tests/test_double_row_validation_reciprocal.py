import geopandas as gpd
from shapely.geometry import LineString
from src.postprocessing.double_row_validation import validate_double_rows

def test_double_row_reciprocal_and_inter_pair():
    # Line 1, 2, 3 as listed in V3 audit:
    # line 1 -> y = 0.00
    # line 2 -> y = 0.90  (0.90m from line 1 - valid intra-pair)
    # line 3 -> y = 2.20  (1.30m from line 2 - valid inter-pair)
    line1 = LineString([(0, 0), (10, 0)])
    line2 = LineString([(0, 0.90), (10, 0.90)])
    line3 = LineString([(0, 2.20), (10, 2.20)])
    
    gdf = gpd.GeoDataFrame(geometry=[line1, line2, line3], crs='EPSG:32630')
    
    config = {
        'postprocessing': {
            'double_row_validation': {
                'enabled': True,
                'mode': 'annotate',
                'intra_pair_min_m': 0.85,
                'intra_pair_max_m': 1.00,
                'inter_pair_min_m': 1.15,
                'inter_pair_max_m': 1.40,
                'tolerance_m': 0.08,
                'max_angle_difference_deg': 8.0,
                'min_longitudinal_overlap_ratio': 0.50
            }
        }
    }
    
    validated = validate_double_rows(gdf, config)
    
    # 1. Line 1 and 2 are matched together as a stable reciprocal pair
    assert validated.loc[0, 'pair_status'] == 'valid_pair'
    assert validated.loc[1, 'pair_status'] == 'valid_pair'
    assert validated.loc[0, 'pair_id'] == validated.loc[1, 'pair_id']
    
    # 2. Line 3 is unmatched (no partner within intra-pair distance)
    assert validated.loc[2, 'pair_status'] == 'unmatched'
    import pandas as pd
    assert pd.isna(validated.loc[2, 'pair_id'])
    
    # 3. Inter-pair neighbor must be successfully detected for Line 2 and Line 3 (distance is 1.30m) (BUG P1-01)
    assert validated.loc[1, 'inter_pair_status'] == 'neighbor_found'
    assert validated.loc[2, 'inter_pair_status'] == 'neighbor_found'
    
    # 4. Line 1 has no inter-pair neighbor (distance to line 3 is 2.20m, exceeding 1.40 + 0.08 = 1.48m)
    assert validated.loc[0, 'inter_pair_status'] == 'no_neighbor'


def test_double_row_filter_mode_preserves_unmatched():
    # Verify that mode='filter' preserves 'unmatched' lines but filters out invalid ones (BUG P1-04)
    line1 = LineString([(0, 0), (10, 0)])
    line2 = LineString([(0, 0.90), (10, 0.90)])
    line3 = LineString([(0, 2.20), (10, 2.20)]) # unmatched
    line4 = LineString([(0, 0.50), (10, 0.50)]) # invalid distance with line 1 (0.50m < 0.85 - 0.08)
    
    gdf = gpd.GeoDataFrame(geometry=[line1, line2, line3, line4], crs='EPSG:32630')
    
    config = {
        'postprocessing': {
            'double_row_validation': {
                'enabled': True,
                'mode': 'filter',
                'intra_pair_min_m': 0.85,
                'intra_pair_max_m': 1.00,
                'inter_pair_min_m': 1.15,
                'inter_pair_max_m': 1.40,
                'tolerance_m': 0.08,
                'max_angle_difference_deg': 8.0,
                'min_longitudinal_overlap_ratio': 0.50
            }
        }
    }
    
    filtered = validate_double_rows(gdf, config)
    
    # unmatched (line 3) is preserved, invalid_distance (line 4) is removed
    # Total remaining lines should be 3 (line 1, line 2, line 3)
    assert len(filtered) == 3
    assert any(filtered.geometry.geom_equals(line3))
    assert not any(filtered.geometry.geom_equals(line4))
