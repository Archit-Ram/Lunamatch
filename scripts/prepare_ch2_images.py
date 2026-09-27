#!/usr/bin/env python3
"""
Find overlapping regions between TMC-2 NCA (Aft) and NCN (Nadir) strip images
using their geometry CSV files, then crop matching patches.
"""

import csv
import os
from PIL import Image

BUNDLE = "/Users/ramarchit/Desktop/Lunamatch-main/ch2_tmc_nca_20211122T2123225689_d_img_d18_Bundle"

NCA_CSV = os.path.join(BUNDLE, "ch2_tmc_nca_20211122T2123225689_d_img_d18/geometry/calibrated/20211122/ch2_tmc_nca_20211122T2123225689_g_grd_d18.csv")
NCN_CSV = os.path.join(BUNDLE, "ch2_tmc_ncn_20211122T2123225689_d_img_d18/geometry/calibrated/20211122/ch2_tmc_ncn_20211122T2123225689_g_grd_d18.csv")

NCA_PNG = os.path.join(BUNDLE, "ch2_tmc_nca_20211122T2123225689_d_img_d18/browse/calibrated/20211122/ch2_tmc_nca_20211122T2123225689_b_brw_d18.png")
NCN_PNG = os.path.join(BUNDLE, "ch2_tmc_ncn_20211122T2123225689_d_img_d18/browse/calibrated/20211122/ch2_tmc_ncn_20211122T2123225689_b_brw_d18.png")

OUTPUT_DIR = "/Users/ramarchit/Desktop/Lunamatch-main/public/test_images"

def read_geometry(csv_path):
    """Read geometry CSV and build a scan-line -> lat/lon mapping."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'lon': float(row['Longitude']),
                'lat': float(row['Latitude']),
                'pixel': int(row['Pixel']),
                'scan': int(row['Scan']),
            })
    return rows

def get_center_lat_by_scan(geo_rows):
    """Get center pixel latitude for each scan line."""
    scan_to_lat = {}
    for row in geo_rows:
        if row['pixel'] == 200:  # Center pixel (images are 400px wide)
            scan_to_lat[row['scan']] = row['lat']
    return scan_to_lat

nca_geo = read_geometry(NCA_CSV)
ncn_geo = read_geometry(NCN_CSV)

nca_scan_lat = get_center_lat_by_scan(nca_geo)
ncn_scan_lat = get_center_lat_by_scan(ncn_geo)

# Get available scan lines
nca_scans = sorted(nca_scan_lat.keys())
ncn_scans = sorted(ncn_scan_lat.keys())

print(f"NCA: {len(nca_scans)} scan lines, lat range: {nca_scan_lat[nca_scans[0]]:.4f} to {nca_scan_lat[nca_scans[-1]]:.4f}")
print(f"NCN: {len(ncn_scans)} scan lines, lat range: {ncn_scan_lat[ncn_scans[0]]:.4f} to {ncn_scan_lat[ncn_scans[-1]]:.4f}")

# Find overlapping latitude range
nca_lat_min = min(nca_scan_lat.values())
nca_lat_max = max(nca_scan_lat.values())
ncn_lat_min = min(ncn_scan_lat.values())
ncn_lat_max = max(ncn_scan_lat.values())

overlap_lat_min = max(nca_lat_min, ncn_lat_min)
overlap_lat_max = min(nca_lat_max, ncn_lat_max)

print(f"\nOverlap latitude range: {overlap_lat_min:.4f} to {overlap_lat_max:.4f}")
print(f"Overlap extent: {abs(overlap_lat_max - overlap_lat_min):.4f}°")

if overlap_lat_min >= overlap_lat_max:
    print("ERROR: No overlap found!")
    exit(1)

# For each target latitude, find the closest scan line in each image
def find_scan_for_lat(scan_lat_dict, target_lat):
    """Find the scan line closest to a target latitude."""
    best_scan = None
    best_dist = float('inf')
    for scan, lat in scan_lat_dict.items():
        dist = abs(lat - target_lat)
        if dist < best_dist:
            best_dist = dist
            best_scan = scan
    return best_scan

# The browse images are subsampled. We need to map from full-res scan lines
# to browse image rows. The browse is 19493 rows. Let's find the ratio.
nca_img = Image.open(NCA_PNG)
ncn_img = Image.open(NCN_PNG)
print(f"\nNCA browse: {nca_img.size[0]}x{nca_img.size[1]}")
print(f"NCN browse: {ncn_img.size[0]}x{ncn_img.size[1]}")

# The scan lines in the CSV go from 0 to max_scan
nca_max_scan = max(nca_scans)
ncn_max_scan = max(ncn_scans)
nca_scale = nca_img.size[1] / nca_max_scan  # browse_rows / full_res_scans
ncn_scale = ncn_img.size[1] / ncn_max_scan

print(f"NCA scale: {nca_scale:.4f} (max scan={nca_max_scan})")
print(f"NCN scale: {ncn_scale:.4f} (max scan={ncn_max_scan})")

# Pick 3 target latitudes within the overlap region
margin = (overlap_lat_max - overlap_lat_min) * 0.15
targets = [
    ("overlap_north", overlap_lat_max - margin),
    ("overlap_center", (overlap_lat_min + overlap_lat_max) / 2),
    ("overlap_south", overlap_lat_min + margin),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
CROP_HEIGHT = 400

for name, target_lat in targets:
    nca_scan = find_scan_for_lat(nca_scan_lat, target_lat)
    ncn_scan = find_scan_for_lat(ncn_scan_lat, target_lat)
    
    nca_browse_row = int(nca_scan * nca_scale)
    ncn_browse_row = int(ncn_scan * ncn_scale)
    
    # Crop 400px tall from each, centered on the target row
    nca_y0 = max(0, nca_browse_row - CROP_HEIGHT // 2)
    nca_y1 = min(nca_img.size[1], nca_y0 + CROP_HEIGHT)
    ncn_y0 = max(0, ncn_browse_row - CROP_HEIGHT // 2)
    ncn_y1 = min(ncn_img.size[1], ncn_y0 + CROP_HEIGHT)
    
    nca_crop = nca_img.crop((0, nca_y0, 400, nca_y1))
    ncn_crop = ncn_img.crop((0, ncn_y0, 400, ncn_y1))
    
    # Resize to 480x480
    nca_resized = nca_crop.resize((480, 480), Image.LANCZOS)
    ncn_resized = ncn_crop.resize((480, 480), Image.LANCZOS)
    
    src_path = os.path.join(OUTPUT_DIR, f"ch2_tmc_nca_{name}.png")
    ref_path = os.path.join(OUTPUT_DIR, f"ch2_tmc_ncn_{name}.png")
    nca_resized.save(src_path)
    ncn_resized.save(ref_path)
    
    print(f"\n--- {name} (lat={target_lat:.4f}°) ---")
    print(f"  NCA scan={nca_scan}, browse row={nca_browse_row}, crop=[{nca_y0}:{nca_y1}]")
    print(f"  NCN scan={ncn_scan}, browse row={ncn_browse_row}, crop=[{ncn_y0}:{ncn_y1}]")
    print(f"  Lat at NCA: {nca_scan_lat[nca_scan]:.4f}°, Lat at NCN: {ncn_scan_lat[ncn_scan]:.4f}°")
    print(f"  → {src_path}")
    print(f"  → {ref_path}")

print(f"\n✅ All overlap-aligned crops saved!")
