import geopandas as gpd
from shapely.geometry import LineString
from src.postprocessing.double_row_validation import group_double_rows, validate_double_rows

def test_double_row_validation():
    # Line 1 and 2 are 0.90m apart (valid double row)
    line1 = LineString([(0, 0), (10, 0)])
    line2 = LineString([(0, 0.90), (10, 0.90)])
    # Line 3 is too close to Line 1 (invalid distance, not a valid double row)
    line3 = LineString([(0, 0.50), (10, 0.50)])
    
    gdf = gpd.GeoDataFrame(geometry=[line1, line2, line3], crs='EPSG:32630')
    
    validated_gdf = group_double_rows(
        gdf,
        intra_pair_min_m=0.85,
        intra_pair_max_m=1.00,
        inter_pair_min_m=1.15,
        inter_pair_max_m=1.40
    )
    
    # Line 1 and Line 2 should be kept, Line 3 should be filtered out
    assert len(validated_gdf) == 2
    assert any(validated_gdf.geometry.geom_equals(line1))
    assert any(validated_gdf.geometry.geom_equals(line2))
    assert not any(validated_gdf.geometry.geom_equals(line3))

def test_validate_double_rows_config():
    line1 = LineString([(0, 0), (10, 0)])
    line2 = LineString([(0, 0.90), (10, 0.90)])
    gdf = gpd.GeoDataFrame(geometry=[line1, line2], crs='EPSG:32630')
    
    # When validation is disabled, all lines should be kept
    config_disabled = {
        'postprocessing': {
            'double_row_validation': {
                'enabled': False
            }
        }
    }
    validated_disabled = validate_double_rows(gdf, config_disabled)
    assert len(validated_disabled) == 2
