import argparse

def get_base_parser() -> argparse.ArgumentParser:
    """
    Returns standard parser with base YAML configuration argument.
    """
    parser = argparse.ArgumentParser(description="Sugarcane Row AI Detection & Vectorization")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/unet_resnet34.yaml',
        help='Path to the model configuration YAML file (relative to root)'
    )
    return parser
