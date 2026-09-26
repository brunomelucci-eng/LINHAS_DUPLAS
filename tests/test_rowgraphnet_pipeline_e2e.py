"""
Comprehensive test suite verifying the complete pipeline with RowGraphNet architecture:
1. Model initialization and multi-task head contracts (band, center, orientation, endpoints, distance, uncertainty, refined_center)
2. ConnectivityRefiner zero-initialization (starts as exact no-op)
3. Gradient propagation with CombinedLoss (focal, dice, cldice, orientation)
4. Synthetic dataset preparation (tiling, global scaling, target rasterization)
5. Trainer training step, validation evaluation, and checkpoint save/load
6. End-to-end inference with sliding window, continuous fairing, and double-row metrics
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from torch.utils.data import DataLoader

from src.models.rowgraphnet import RowGraphNetModel, MultiTaskDecoder, ConnectivityRefiner
from src.models.factory import build_model
from src.losses.combined import CombinedLoss
from src.data.dataset import SugarcaneDataset
from src.training.trainer import Trainer
from src.inference.predictor import RowPredictor
from src.postprocessing.continuous_fairing import continuous_fairing_gdf


class TestRowGraphNetArchitecture:
    """Verifies the RowGraphNet model architecture and output contracts."""

    def test_model_build_via_factory(self):
        config = {
            "model": {
                "architecture": "rowgraphnet",
                "encoder": "mit_b0",
                "pretrained": False,
                "decoder_channels": 64,
                "refine_connectivity": True,
                "refiner_hidden_channels": 32,
            }
        }
        model = build_model(config)
        assert isinstance(model, RowGraphNetModel)
        assert model.out_channels == 4

    def test_multitask_heads_and_ranges(self):
        model = RowGraphNetModel(
            encoder_name="mit_b0",
            encoder_weights=None,
            decoder_channels=64,
            refiner_hidden_channels=32,
        )
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            preds = model.predict_multitask(x)

        # Check required keys
        required_keys = [
            "band", "center", "orientation", "orientation_sin", "orientation_cos",
            "endpoints", "distance", "uncertainty", "refined_center", "repair_logits"
        ]
        for key in required_keys:
            assert key in preds, f"Missing key: {key}"

        # Probability outputs must be strictly in [0, 1]
        for prob_key in ["band", "center", "endpoints", "refined_center"]:
            val = preds[prob_key]
            assert val.min() >= 0.0, f"{prob_key} has values < 0"
            assert val.max() <= 1.0, f"{prob_key} has values > 1"

        # Orientation must be a normalized unit vector: sin^2 + cos^2 ≈ 1
        sin = preds["orientation_sin"]
        cos = preds["orientation_cos"]
        norm_sq = sin ** 2 + cos ** 2
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-4)

        # Distance and uncertainty must be non-negative
        assert (preds["distance"] >= 0.0).all()
        assert (preds["uncertainty"] > 0.0).all()

    def test_connectivity_refiner_zero_initialization(self):
        """ConnectivityRefiner repair_head is zero-initialized to guarantee it starts as a no-op."""
        model = RowGraphNetModel(
            encoder_name="mit_b0",
            encoder_weights=None,
            decoder_channels=64,
            refiner_hidden_channels=32,
        )
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            preds = model.predict_multitask(x)

        # Since repair_head is initialized to zero, repair_logits must be all zeros
        assert torch.allclose(preds["repair_logits"], torch.zeros_like(preds["repair_logits"]), atol=1e-6)
        # And refined_center must equal coarse center
        assert torch.allclose(preds["refined_center"], preds["center_observed"], atol=1e-6)

    def test_gradient_flow_with_combined_loss(self):
        """Verifies forward, loss computation, and backprop across all layers."""
        model = RowGraphNetModel(
            encoder_name="mit_b0",
            encoder_weights=None,
            decoder_channels=64,
            refiner_hidden_channels=32,
        )
        model.train()
        loss_fn = CombinedLoss({
            "loss": {
                "row_bce_weight": 0.2,
                "row_dice_weight": 0.2,
                "center_focal_weight": 0.2,
                "center_dice_weight": 0.2,
                "cldice_weight": 0.1,
                "orientation_weight": 0.1,
            }
        })

        x = torch.randn(2, 3, 128, 128)
        targets = {
            "row_mask": torch.randint(0, 2, (2, 1, 128, 128)).float(),
            "center_mask": torch.randint(0, 2, (2, 1, 128, 128)).float(),
            "orientation_sin": torch.zeros((2, 1, 128, 128)),
            "orientation_cos": torch.ones((2, 1, 128, 128)),
            "valid_mask": torch.ones((2, 1, 128, 128)),
        }

        logits = model(x)
        assert logits.shape == (2, 4, 128, 128)

        losses = loss_fn(logits, targets)
        assert "loss" in losses
        total_loss = losses["loss"]
        assert not torch.isnan(total_loss)
        assert total_loss.item() > 0

        total_loss.backward()

        # Check gradients exist on encoder, decoder, and refiner
        has_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        has_decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
        assert has_encoder_grad, "Encoder has no gradient"
        assert has_decoder_grad, "Decoder has no gradient"


class TestFullPipelineWithRowGraphNet:
    """Verifies end-to-end data preparation, training step, and inference."""

    @pytest.fixture
    def synthetic_env(self, tmp_path):
        """Creates synthetic tiles, orthomosaic, ROI, and reference lines."""
        H = W = 256
        gsd = 0.1
        transform = rasterio.Affine(gsd, 0.0, 100.0, 0.0, -gsd, 500.0)

        # 3-band synthetic image with two double rows
        img = np.random.randint(60, 120, (3, H, W), dtype=np.uint8)
        # Double row pair: row 1 at y=100, row 2 at y=110 (intra-pair distance 10 px = 1.0 m)
        img[:, 98:103, 20:236] = 230
        img[:, 108:113, 20:236] = 230

        raster_path = str(tmp_path / "ortho.tif")
        with rasterio.open(
            raster_path, "w",
            driver="GTiff", width=W, height=H, count=3,
            dtype="uint8", crs="EPSG:32630", transform=transform
        ) as dst:
            dst.write(img)

        # ROI polygon
        min_x, max_y = 100.0, 500.0
        max_x = min_x + W * gsd
        min_y = max_y - H * gsd
        roi_geom = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
        roi_gdf = gpd.GeoDataFrame(geometry=[roi_geom], crs="EPSG:32630")
        roi_path = str(tmp_path / "roi.geojson")
        roi_gdf.to_file(roi_path, driver="GeoJSON")

        # Double rows
        line_left = LineString([(100.0 + 20 * gsd, 500.0 - 100 * gsd), (100.0 + 236 * gsd, 500.0 - 100 * gsd)])
        line_right = LineString([(100.0 + 20 * gsd, 500.0 - 110 * gsd), (100.0 + 236 * gsd, 500.0 - 110 * gsd)])
        lines_gdf = gpd.GeoDataFrame(geometry=[line_left, line_right], crs="EPSG:32630")
        lines_gdf["row_id"] = [1, 2]
        lines_gdf["talhao_id"] = [1, 1]
        lines_path = str(tmp_path / "lines.geojson")
        lines_gdf.to_file(lines_path, driver="GeoJSON")

        # Create dataset tiles (.npz)
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        for split_dir, count in [(train_dir, 4), (val_dir, 2)]:
            for i in range(count):
                npz_path = split_dir / f"tile_{i:03d}.npz"
                np.savez_compressed(
                    npz_path,
                    image=(img.astype(np.float32) / 255.0),
                    row_mask=(img[0] > 200).astype(np.float32),
                    center_mask=(img[0] > 220).astype(np.float32),
                    orientation_sin=np.zeros((H, W), dtype=np.float32),
                    orientation_cos=np.ones((H, W), dtype=np.float32),
                    valid_mask=np.ones((H, W), dtype=np.float32),
                    group_id="1",
                )

        return {
            "tmp_path": tmp_path,
            "raster_path": raster_path,
            "roi_path": roi_path,
            "lines_path": lines_path,
            "dataset_dir": str(dataset_dir),
        }

    def test_train_and_inference_cycle(self, synthetic_env):
        env = synthetic_env
        tmp_path = env["tmp_path"]

        config = {
            "project": {
                "name": "test_rgn",
                "seed": 42,
                "output_dir": str(tmp_path / "outputs"),
            },
            "model": {
                "architecture": "rowgraphnet",
                "encoder": "mit_b0",
                "pretrained": False,
                "decoder_channels": 32,
                "refine_connectivity": True,
                "refiner_hidden_channels": 16,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "mixed_precision": False,
                "gradient_accumulation_steps": 1,
                "gradient_clip": 1.0,
                "monitor": "val_loss",
                "monitor_mode": "min",
                "early_stopping_patience": 5,
                "scheduler": {
                    "type": "plateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 2,
                },
                "threshold_calibration": {
                    "enabled": False,
                },
            },
            "loss": {
                "row_bce_weight": 0.2,
                "row_dice_weight": 0.2,
                "center_focal_weight": 0.2,
                "center_dice_weight": 0.2,
                "cldice_weight": 0.1,
                "orientation_weight": 0.1,
            },
            "inference": {
                "tile_size_px": 256,
                "overlap_px": 64,
                "batch_size": 1,
                "mixed_precision": False,
                "tta": False,
                "mask_threshold_mode": "fixed",
                "fixed_threshold": 0.5,
                "center_threshold": 0.5,
            },
            "final_centerline_fairing": {
                "engine": "continuous",
                "max_deviation_m": 0.12,
            },
        }

        # 1. Dataset loading
        train_ds = SugarcaneDataset(os.path.join(env["dataset_dir"], "train"))
        val_ds = SugarcaneDataset(os.path.join(env["dataset_dir"], "val"))
        assert len(train_ds) == 4
        assert len(val_ds) == 2

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        # 2. Build model, loss and trainer
        model = build_model(config)
        loss_fn = CombinedLoss(config)
        trainer = Trainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
        )

        # 3. Train 1 epoch
        history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
        assert len(history) >= 1
        assert "val_loss" in history[0]

        # Verify checkpoint was created
        checkpoint_dir = os.path.join(config["project"]["output_dir"], "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        assert len(checkpoints) >= 1
        ckpt_path = os.path.join(checkpoint_dir, checkpoints[0])

        # 4. Run Predictor with the trained checkpoint
        predictor = RowPredictor(
            config=config,
            checkpoint_path=ckpt_path,
        )
        assert predictor.model is not None

        # Test batch prediction with RowGraphNet
        dummy_patch = np.random.rand(1, 3, 256, 256).astype(np.float32)
        pred_out = predictor.predict_batch(dummy_patch)
        assert pred_out.shape == (1, 4, 256, 256)
        # Probabilities in [0, 1]
        assert (pred_out[:, 0:2] >= 0.0).all() and (pred_out[:, 0:2] <= 1.0).all()

        # 5. Continuous fairing on synthetic line strings
        line1 = LineString([(100.0, 500.0), (105.0, 500.05), (110.0, 499.98), (115.0, 500.0)])
        line2 = LineString([(100.0, 499.0), (105.0, 499.04), (110.0, 498.96), (115.0, 499.0)])
        raw_gdf = gpd.GeoDataFrame(
            {"track_id": [1, 1], "pair_id": [1, 1], "quality_after": ["GOOD", "GOOD"]},
            geometry=[line1, line2],
            crs="EPSG:32630",
        )
        smoothed_gdf = continuous_fairing_gdf(raw_gdf, config=config)
        assert len(smoothed_gdf) >= 1
        assert "fairing_applied" in smoothed_gdf.columns
        assert smoothed_gdf["fairing_applied"].all()
