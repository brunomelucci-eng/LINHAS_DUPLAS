import os
import json
import logging

logger = logging.getLogger(__name__)

def generate_html_report(output_dir: str):
    """
    Generate an HTML file linking metrics and images for visual inspection.
    """
    report_json_path = os.path.join(output_dir, 'validation_report.json')
    if not os.path.exists(report_json_path):
        logger.warning(f"Could not generate HTML report: {report_json_path} does not exist.")
        return
        
    with open(report_json_path, 'r') as f:
        metrics = json.load(f)
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sugarcane Row AI - Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f4f6f8;
            color: #333;
        }}
        h1 {{ color: #1b5e20; border-bottom: 3px solid #81c784; padding-bottom: 10px; }}
        h2 {{ color: #37474f; border-bottom: 2px solid #cfd8dc; padding-bottom: 5px; margin-top: 30px; }}
        .metric-card {{
            background: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            display: inline-block;
            min-width: 220px;
            vertical-align: top;
        }}
        .metric-value {{ font-size: 26px; font-weight: bold; color: #2e7d32; }}
        .metric-title {{ font-size: 14px; color: #78909c; margin-top: 5px; }}
        .grid {{ display: flex; flex-wrap: wrap; }}
        .img-container {{
            margin: 15px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            flex: 1;
            min-width: 400px;
        }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 10px auto; }}
        h3 {{ text-align: center; color: #455a64; }}
    </style>
</head>
<body>
    <h1>Sugarcane Row Detection & Vectorization Report</h1>
    <p>Pipeline execution successfully completed. Review evaluation results below.</p>
    
    <h2>Segmentation Performance</h2>
    <div class="grid">
        <div class="metric-card">
            <div class="metric-value">{metrics['segmentation'].get('dice', 0.0):.4f}</div>
            <div class="metric-title">Row Dice Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['segmentation'].get('iou', 0.0):.4f}</div>
            <div class="metric-title">Row IoU</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['segmentation'].get('cldice', 0.0):.4f}</div>
            <div class="metric-title">clDice (Centerline) Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['segmentation'].get('precision', 0.0):.4f}</div>
            <div class="metric-title">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['segmentation'].get('recall', 0.0):.4f}</div>
            <div class="metric-title">Recall</div>
        </div>
    </div>
    
    <h2>Geometric Alignment</h2>
    <div class="grid">
        <div class="metric-card">
            <div class="metric-value">{metrics['geometry'].get('mean_symmetric_distance_m', float('nan')):.4f} m</div>
            <div class="metric-title">Mean Symmetric Distance</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['geometry'].get('hausdorff_distance_m', float('nan')):.4f} m</div>
            <div class="metric-title">Hausdorff Distance</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['geometry'].get('hd95_m', float('nan')):.4f} m</div>
            <div class="metric-title">HD95 Distance</div>
        </div>
    </div>

    <h2>Topological Summary</h2>
    <div class="grid">
        <div class="metric-card">
            <div class="metric-value">{metrics['topology'].get('num_reconstructed_lines', 0)}</div>
            <div class="metric-title">Reconstructed Row Lines</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['topology'].get('num_crossover_intersections', 0)}</div>
            <div class="metric-title">Crossover Crossing Points</div>
        </div>
    </div>
    
    <h2>Visual Diagnostic Artifacts</h2>
    <div class="grid">
        <div class="img-container">
            <h3>Predicted Rows Overlay</h3>
            <img src="overlays.png" alt="Overlay Plot">
        </div>
        <div class="img-container">
            <h3>Probability Heatmaps</h3>
            <img src="probabilities.png" alt="Probability Heatmap">
        </div>
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, 'summary_report.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    logger.info(f"HTML summary report generated at {html_path}")
