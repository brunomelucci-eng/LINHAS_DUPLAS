import yaml
import os
from typing import Optional

def load_config(config_path: Optional[str], base_config_path: str = "configs/base.yaml") -> dict:
    """
    Loads base config and merges user configuration overlay parameters into it.
    """
    # Robust path resolution
    resolved_base = base_config_path
    if not os.path.exists(resolved_base):
        resolved_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_config_path)
        if not os.path.exists(resolved_base):
            raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")
            
    with open(resolved_base, 'r') as f:
        config = yaml.safe_load(f)
        
    if config_path:
        resolved_user = config_path
        if not os.path.exists(resolved_user):
            resolved_user = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
            if not os.path.exists(resolved_user):
                raise FileNotFoundError(f"User configuration file not found: {config_path}")
                
        with open(resolved_user, 'r') as f:
            user_config = yaml.safe_load(f)
            
        def merge(dict1, dict2):
            for k, v in dict2.items():
                if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
                    merge(dict1[k], v)
                else:
                    dict1[k] = v
                    
        if user_config:
            merge(config, user_config)
            
    return config
