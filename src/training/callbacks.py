import logging
import os
import csv
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CSVMetricLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, 'metrics.csv')
        self.initialized = False

    def log(self, metrics: Dict[str, Any]):
        os.makedirs(self.output_dir, exist_ok=True)
        mode = 'a' if self.initialized and os.path.exists(self.csv_path) else 'w'
        
        keys = list(metrics.keys())
        if 'epoch' in keys:
            keys.remove('epoch')
            keys = ['epoch'] + sorted(keys)
        else:
            keys = sorted(keys)
            
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if mode == 'w':
                writer.writeheader()
            writer.writerow(metrics)
            
        self.initialized = True
