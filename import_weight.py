from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

# Create output directories
eo_dir = Path("data/Prithvi-EO")
wxc_dir = Path("data/Prithvi-WxC")

eo_dir.mkdir(parents=True, exist_ok=True)
wxc_dir.mkdir(parents=True, exist_ok=True)

# Download Prithvi-EO model
hf_hub_download(
    repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    filename="config.json",
    local_dir=eo_dir,
)


hf_hub_download(
    repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    filename="Prithvi_EO_V2_300M.pt",
    local_dir=eo_dir,
)


# Download Prithvi-WxC model — use different approach
print("Downloading Prithvi-WxC model...")
hf_hub_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    filename="config.yaml",
    local_dir=wxc_dir,
)

hf_hub_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    filename="prithvi.wxc.2300m.v1.pt",
    local_dir=wxc_dir,
)

print(f"✓ Saved to {wxc_dir}/")

print("\n✓ All downloads complete!")