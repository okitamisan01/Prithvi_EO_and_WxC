from time import sleep
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PrithviWxC.download import get_prithvi_wxc_input


import sys, os
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
#  TARGET SETTINGS 
# ════════════════════════════════════════════════════════════════════════════════

# TARGET_GEOIDS = ["01001", "01003"]   # list of GEOID which wanna forcus
TARGET_GEOIDS = [
    # ── Major Agricultural Counties ──────────────────────────────────────────
    "06029",  # Kern, CA          (grapes, citrus, cotton)
    "06019",  # Fresno, CA        (almonds, grapes, tomatoes)
    "06077",  # San Joaquin, CA   (wine grapes, cherries)
    "06047",  # Merced, CA        (dairy, almonds)
    "06039",  # Madera, CA        (grapes, poultry)
    "06031",  # Kings, CA         (cotton, tomatoes)
    "06107",  # Tulare, CA        (dairy, citrus)
    "06099",  # Stanislaus, CA    (dairy, walnuts)
    "17019",  # Champaign, IL     (corn, soybeans)
    "17113",  # McLean, IL        (corn, soybeans)
    "17203",  # Woodford, IL      (corn, soybeans)
    "19153",  # Polk, IA          (corn, soybeans)
    "19013",  # Black Hawk, IA    (corn, hogs)
    "19193",  # Woodbury, IA      (corn, cattle)
    "20015",  # Butler, KS        (wheat, cattle)
    "20055",  # Finney, KS        (feedlots, wheat)
    "20057",  # Ford, KS          (beef cattle, wheat)
    "20091",  # Johnson, KS       (wheat, corn)
    "27013",  # Blue Earth, MN    (corn, soybeans)
    "27103",  # Nicollet, MN      (corn, soybeans)
    "28001",  # Adams, MS         (cotton, soybeans)
    "28027",  # Coahoma, MS       (cotton, rice)
    "28083",  # Leflore, MS       (cotton, soybeans)
    "29037",  # Cass, MO          (corn, soybeans)
    "31153",  # Sarpy, NE         (corn, soybeans)
    "31055",  # Douglas, NE       (corn, cattle)
    "31109",  # Lancaster, NE     (corn, soybeans)
    "35025",  # Lea, NM           (beef cattle, oil)
    "36067",  # Onondaga, NY      (dairy, apples)
    "37067",  # Forsyth, NC       (tobacco, sweet potatoes)
    "38015",  # Burleigh, ND      (wheat, sunflowers)
    "38017",  # Cass, ND          (wheat, soybeans)
    "39021",  # Champaign, OH     (corn, soybeans)
    "40109",  # Oklahoma, OK      (wheat, cattle)
    "41067",  # Washington, OR    (nursery, wheat)
    "42041",  # Cumberland, PA    (dairy, mushrooms)
    "45063",  # Lexington, SC     (peaches, poultry)
    "46099",  # Minnehaha, SD     (corn, soybeans)
    "47149",  # Rutherford, TN    (soybeans, tobacco)
    "48139",  # Ellis, TX         (cotton, wheat)
    "48269",  # King, TX          (beef cattle, cotton)
    "48381",  # Randall, TX       (feedlots, wheat)
    "48421",  # Sherman, TX       (wheat, feedlots)
    "48111",  # Dawson, TX        (cotton, sorghum)
    "49005",  # Cache, UT         (dairy, hay)
    "53077",  # Yakima, WA        (apples, hops)
    "53071",  # Walla Walla, WA   (wheat, wine grapes)
    "55025",  # Dane, WI          (dairy, corn)
    "55087",  # Polk, WI          (dairy, potatoes)
    "56021",  # Laramie, WY       (beef cattle, wheat)

    # ── Major Urban / Metro Counties ─────────────────────────────────────────
    "06037",  # Los Angeles, CA
    "06073",  # San Diego, CA
    "06075",  # San Francisco, CA
    "06085",  # Santa Clara, CA   (Silicon Valley)
    "08031",  # Denver, CO
    "09003",  # Hartford, CT
    "11001",  # Washington, DC
    "12086",  # Miami-Dade, FL
    "12057",  # Hillsborough, FL  (Tampa)
    "12095",  # Orange, FL        (Orlando)
    "13121",  # Fulton, GA        (Atlanta)
    "15003",  # Honolulu, HI
    "17031",  # Cook, IL          (Chicago)
    "18097",  # Marion, IN        (Indianapolis)
    "21111",  # Jefferson, KY     (Louisville)
    "22071",  # Orleans, LA       (New Orleans)
    "22033",  # East Baton Rouge, LA
    "25025",  # Suffolk, MA       (Boston)
    "24510",  # Baltimore City, MD
    "26163",  # Wayne, MI         (Detroit)
    "27053",  # Hennepin, MN      (Minneapolis)
    "28049",  # Hinds, MS         (Jackson)
    "29095",  # Jackson, MO       (Kansas City)
    "29189",  # St. Louis, MO
    "32003",  # Clark, NV         (Las Vegas)
    "33011",  # Hillsborough, NH  (Manchester)
    "34013",  # Essex, NJ         (Newark)
    "35001",  # Bernalillo, NM    (Albuquerque)
    "36005",  # Bronx, NY
    "36047",  # Kings, NY         (Brooklyn)
    "36061",  # New York, NY      (Manhattan)
    "36081",  # Queens, NY
    "37119",  # Mecklenburg, NC   (Charlotte)
    "39035",  # Cuyahoga, OH      (Cleveland)
    "39049",  # Franklin, OH      (Columbus)
    "39061",  # Hamilton, OH      (Cincinnati)
    "40143",  # Tulsa, OK
    "41051",  # Multnomah, OR     (Portland)
    "42101",  # Philadelphia, PA
    "44007",  # Providence, RI
    "45045",  # Greenville, SC
    "47037",  # Davidson, TN      (Nashville)
    "47157",  # Shelby, TN        (Memphis)
    "48029",  # Bexar, TX         (San Antonio)
    "48113",  # Dallas, TX
    "48201",  # Harris, TX        (Houston)
    "48453",  # Travis, TX        (Austin)
    "49035",  # Salt Lake, UT
    "51760",  # Richmond City, VA
    "51059",  # Fairfax, VA
    "53033",  # King, WA          (Seattle)
    "55079",  # Milwaukee, WI
]
TARGET_YEAR_EO   = 2020                  # year
# TARGET_CROP   = "CORN"                # "CORN" / "SOYBEANS" / "WHEAT" (etc.)

# ════════════════════════════════════════════════════════════════════════════════

# ── Submodule paths ────────────────────────────────────────────────────────────
REPO_ROOT   = Path("C:/Users/room208/mizuho")
EO_DIR      = REPO_ROOT / "Prithvi-EO-2.0-300M"
WXC_DIR     = REPO_ROOT / "Prithvi-WxC"
DATA_DIR    = REPO_ROOT / "data"
OUTPUT_DIR  = DATA_DIR   / "mizuho_output"


# ── Configuration ─────────────────────────────────────────────────────────────
YEAR_WxC         = 2020
MONTH        = 1    # January
INPUT_STEP   = 6    # hours between the two input snapshots
LEAD_TIME    = 12   # max forecast horizon in hours (covers both 6h and 12h forecasts)
MERRA_DIR     = DATA_DIR / "/merra-2"
RAW_DIR = DATA_DIR / "/raw"
CLIM_DIR     = DATA_DIR / "/climatology"
# ──────────────────────────────────────────────────────────────────────────────



for p in [EO_DIR, WXC_DIR]:
    assert p.exists(), f"Submodule not found: {p}"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Common imports ─────────────────────────────────────────────────────────────
import gc
import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── getting information of target centroids based on TARGET_GEOIDS  ─────────────────────────────────
_df_gaz = pd.read_csv(
    DATA_DIR / "2025_Gaz_counties_national.txt",
    sep="|", dtype={"GEOID": str}
)
_df_gaz.columns = _df_gaz.columns.str.strip()
_df_gaz = _df_gaz[["GEOID", "NAME", "USPS", "INTPTLAT", "INTPTLONG"]].rename(
    columns={"INTPTLAT": "lat", "INTPTLONG": "lon", "USPS": "state"}
)

df_target = _df_gaz[_df_gaz["GEOID"].isin(TARGET_GEOIDS)].reset_index(drop=True)
assert len(df_target) > 0, f"GEOID not found: {TARGET_GEOIDS}"

RESOLVED_GEOIDS = df_target["GEOID"].tolist()

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    gc.collect()
    torch.cuda.empty_cache()
print(f"Device  : {device}")
print(f"Target Counties : {list(zip(RESOLVED_GEOIDS, df_target['NAME'].tolist()))}")
print("Submodule paths OK.")



# ── Step 1: Climatology (doy001–doy031 for January) ───────────────────────────
print("Downloading climatology files for January (doy001–doy031)...")
clim_patterns = [f"climatology/climate_surface_doy{d:03d}*.nc" for d in range(1, 32)]
snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    allow_patterns=clim_patterns,
    local_dir= DATA_DIR
)
clim_patterns_vert = [f"climatology/climate_vertical_doy{d:03d}*.nc" for d in range(1, 32)]
snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    allow_patterns=clim_patterns_vert,
    local_dir= DATA_DIR
)
surf_clim_dir = Path(CLIM_DIR)
vert_clim_dir = Path(CLIM_DIR)
print("Climatology download complete.\n")

# ── Step 2: MERRA-2 data (one day at a time) ──────────────────────────────────
start = np.datetime64(f"{YEAR_WxC:04d}-{MONTH:02d}-01")
end   = (
    np.datetime64(f"{YEAR_WxC:04d}-{MONTH+1:02d}-01")
    if MONTH < 12
    else np.datetime64(f"{YEAR_WxC+1:04d}-01-01")
)
days_in_month = int((end - start).astype(int))
dates = [start + np.timedelta64(d, 'D') for d in range(days_in_month)]

print(f"Downloading {len(dates)} days ({YEAR_WxC}-{MONTH:02d}) with lead_time={LEAD_TIME}h ...")

for date in dates:
    date_str = str(date)[:10]
    marker = Path(MERRA_DIR) / f".done_{date_str}"
    if marker.exists():
        print(f"  {date_str}: already on disk, skipping.")
        continue

    print(f"  {date_str}: downloading...", end=" ", flush=True)
    for tries in range(5):
        try:
            get_prithvi_wxc_input(
                date,
                input_time_step=INPUT_STEP,
                lead_time=LEAD_TIME,
                input_data_dir=MERRA_DIR,   # "./data/merra-2"  ← processed input data
                download_dir=RAW_DIR,       # "./data/raw"      ← raw downloaded files
            )
            marker.touch()
            print("done.")
            break
        except Exception as exc:
            print(f"\n    Attempt #{tries} failed: {exc.args[0]}")
            if tries < 4:
                wtime = 2 ** tries
                print(f"    Retrying in {wtime}s...", end=" ", flush=True)
                sleep(wtime)
            else:
                print(f"    Giving up on {date_str}. Re-run to retry.")

print("\nAll downloads complete.")