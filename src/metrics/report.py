import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compile_metrics_report(
    seg_metrics: Dict[str, float],
    geom_metrics: Dict[str, float],
    topo_metrics: Dict[str, int],
    output_dir: str
) -> Dict[str, Any]:
    """
    Compile segmentation, geometry, and topology metrics into a single dictionary,
    log the summary, and write to validation_report.json.
    """
    report = {
        'segmentation': seg_metrics,
        'geometry': geom_metrics,
        'topology': topo_metrics
    }
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'validation_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Validation metrics report successfully saved to {report_path}")
    
    logger.info("=== METRICS REPORT SUMMARY ===")
    logger.info(f"Row Mask Dice:           {seg_metrics.get('dice', 0.0):.4f}")
    logger.info(f"Row Mask IoU:            {seg_metrics.get('iou', 0.0):.4f}")
    logger.info(f"clDice Score:            {seg_metrics.get('cldice', 0.0):.4f}")
    logger.info(f"Mean Symmetric Distance: {geom_metrics.get('mean_symmetric_distance_m', float('nan')):.4f} m")
    logger.info(f"Hausdorff Distance:       {geom_metrics.get('hausdorff_distance_m', float('nan')):.4f} m")
    logger.info(f"HD95 Distance:            {geom_metrics.get('hd95_m', float('nan')):.4f} m")
    logger.info(f"Reconstructed Rows:       {topo_metrics.get('num_reconstructed_lines', 0)}")
    logger.info(f"Crossover Crossings:     {topo_metrics.get('num_crossover_intersections', 0)}")
    logger.info("==============================")
    
    return report
