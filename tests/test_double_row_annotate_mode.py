import geopandas as gpd
from shapely.geometry import LineString
from src.postprocessing.double_row_validation import validate_double_rows

def test_double_row_annotate_mode():
    # Line 1 and 2 are 0.90m apart (valid double row)
    line1 = LineString([(0, 0), (10, 0)])
    line2 = LineString([(0, 0.90), (10, 0.90)])
    # Line 3 is 2.0m away (invalid double row partner)
    line3 = LineString([(0, 2.90), (10, 2.90)])
    
    gdf = gpd.GeoDataFrame(geometry=[line1, line2, line3], crs='EPSG:32630')
    
    # Enable double row validation in annotate mode
    config = {
        'postprocessing': {
            'double_row_validation': {
                'enabled': True,
                'mode': 'annotate',
                'intra_pair_min_m': 0.85,
                'intra_pair_max_m': 1.00,
                'inter_pair_min_m': 1.15,
                'inter_pair_max_m': 1.40,
                'tolerance_m': 0.08
            }
        }
    }
    
    annotated_gdf = validate_double_rows(gdf, config)
    
    # In annotate mode, no lines should be removed
    assert len(annotated_gdf) == 3
    
    # Check that annotation columns are added
    expected_cols = ['pair_id', 'pair_status', 'pair_distance_m', 'pair_angle_deg',
                     'pair_overlap_ratio', 'pair_confidence', 'inter_pair_status']
    for col in expected_cols:
        assert col in annotated_gdf.columns
        
    # Check annotations of first two lines
    assert annotated_gdf.loc[0, 'pair_status'] == 'valid_pair'
    assert annotated_gdf.loc[1, 'pair_status'] == 'valid_pair'
    assert annotated_gdf.loc[2, 'pair_status'] == 'unmatched'
