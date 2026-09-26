from .unet import RowUNet
from .deeplab import RowDeepLab
from .segformer import RowSegFormer
from .rowgraphnet import RowGraphNetModel
from .base import BaseRowModel

def build_model(config: dict) -> BaseRowModel:
    model_cfg = config.get('model', {})
    arch = model_cfg.get('architecture', 'unet').lower()
    default_encoder = 'mit_b0' if arch in ('rowgraphnet', 'segformer') else 'resnet34'
    encoder = model_cfg.get('encoder', default_encoder)
    pretrained = model_cfg.get('pretrained', True)
    in_channels = model_cfg.get('input_channels', 3)
    out_channels = model_cfg.get('output_channels', 4)
    decoder_channels = model_cfg.get('decoder_channels', 128)
    refine_connectivity = model_cfg.get('refine_connectivity', True)
    refiner_hidden_channels = model_cfg.get('refiner_hidden_channels', 64)
    
    weights = 'imagenet' if pretrained else None
    
    if arch == 'rowgraphnet':
        return RowGraphNetModel(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            refine_connectivity=refine_connectivity,
            refiner_hidden_channels=refiner_hidden_channels,
            out_channels=out_channels,
        )
    elif arch == 'unet':
        return RowUNet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif arch == 'deeplab':
        return RowDeepLab(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif arch == 'segformer':
        return RowSegFormer(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            out_channels=out_channels
        )
    else:
        raise ValueError(f"Unknown model architecture: '{arch}'. Choose 'rowgraphnet', 'unet', 'deeplab', or 'segformer'.")
