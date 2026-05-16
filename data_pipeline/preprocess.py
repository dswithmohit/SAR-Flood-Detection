import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

def load_geotiff(file_path: str | Path):
    """Load first band of GeoTIFF and return data + profile."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)   # Ensure float32
        profile = src.profile
    return data, profile

def simple_speckle_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Very basic mean filter as speckle reduction (can be improved)."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(data, size=kernel_size, mode='reflect')

def create_dataset(config=None):
    config = config or {}
    data_dir = Path("data")
    pre_path = data_dir / "raw" / "pre_flood.tif"
    post_path = data_dir / "raw" / "post_flood.tif"
    threshold_db = config.get("water_threshold_db", -16.0)

    print("Starting preprocessing for SAR Flood Detection...")

    if not (pre_path.exists() and post_path.exists()):
        raise FileNotFoundError(f"Raw files not found! Expected at:\n{pre_path}\n{post_path}")

    # Load images
    pre_flood, profile = load_geotiff(pre_path)
    post_flood, _ = load_geotiff(post_path)

    print(f"Image shape: {pre_flood.shape} | Pre range: [{pre_flood.min():.2f}, {pre_flood.max():.2f}] dB")

    # Optional light speckle filtering
    # pre_flood = simple_speckle_filter(pre_flood)
    # post_flood = simple_speckle_filter(post_flood)

    # Core features
    difference = post_flood - pre_flood

    # Automated labels (pseudo ground truth)
    labels = np.where(post_flood < threshold_db, 1, 0).astype(np.uint8)

    # Flatten
    X_pre = pre_flood.flatten()
    X_post = post_flood.flatten()
    X_diff = difference.flatten()
    y = labels.flatten()

    # Clean invalid values
    valid_mask = (
        ~np.isnan(X_pre) & 
        ~np.isnan(X_post) & 
        ~np.isinf(X_pre) & 
        ~np.isinf(X_post)
    )

    X_combined = np.column_stack((
        X_pre[valid_mask], 
        X_post[valid_mask], 
        X_diff[valid_mask]
    ))
    y_combined = y[valid_mask]

    # Save processed data
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_features.npy", X_combined)
    np.save(processed_dir / "y_labels.npy", y_combined)

    # Save label mask as GeoTIFF (very useful for mapping later)
    profile.update(dtype="uint8", count=1, nodata=0) # Fixed dtype string representation
    with rasterio.open(processed_dir / "flood_label_mask.tif", "w", **profile) as dst:
        dst.write(labels, 1)

    # Optional: Save processed pre/post for visualization
    profile.update(dtype="float32", count=1) # Fixed dtype string representation
    with rasterio.open(processed_dir / "pre_flood_processed.tif", "w", **profile) as dst:
        dst.write(pre_flood, 1)
    with rasterio.open(processed_dir / "post_flood_processed.tif", "w", **profile) as dst:
        dst.write(post_flood, 1)

    print(f" Preprocessing complete!")
    print(f"    Features shape : {X_combined.shape}")
    print(f"    Water pixels   : {y_combined.sum():,} ({y_combined.mean()*100:.2f}%)")
    print(f"    Saved to       : {processed_dir}")

if __name__ == "__main__":
    create_dataset()