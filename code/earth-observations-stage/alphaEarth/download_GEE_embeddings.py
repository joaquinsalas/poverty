#import modules
import numpy as np
import pandas as pd
import geopandas as gpd
import geemap
from datetime import datetime
from tqdm import tqdm
import ee
import os
import math

#====================================================================
#initialize Earth Engine
try:
    ee.Initialize(project='ee-salas')
    print("Authenticated successfully")
except Exception as e:
    print("Authentication failed, retrying...")
    ee.Authenticate()
    ee.Initialize()

#====================================================================
#read INEGI grid csv file
filename = '../INEGI_CPV2020_n9/INEGI_CPV2020_n9_.csv'
df_csv = pd.read_csv(filename)
code = df_csv['CODIGO']

#====================================================================
#read data file
path = '../data/'
filename = 'ensemble_inferences_calidad_vivienda_2020.csv'
df_ref = pd.read_csv(f'{path}{filename}')
y_ref = df_ref[[f"prediction_{i:02d}" for i in range(1, 31)]].mean(axis=1)
code_ref = df_ref['codigo']

#====================================================================
#read the grid shapefile
filename = '../INEGI_CPV2020_n9/INEGI_CPV2020_n9_.shp'
gdf = gpd.read_file(filename)

#====================================================================
# Merge shapefile with reference codes (avoids index mismatches)
gdf = gdf.merge(df_ref, left_on='CODIGO', right_on='codigo')

#====================================================================
# Prepare output folder
img_path = '/mnt/data-r1/data/alphaEarth/'
os.makedirs(img_path, exist_ok=True)

#====================================================================
# Define parameters
collection = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
pixel_size_m = 10  # meters per pixel
half_size_pixels = 48
total_size_pixels = half_size_pixels * 2
start_date = '2020-01-01'
end_date = '2020-12-31'


# Make sure centroids are lon/lat
if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading images"):
    code_val = row['CODIGO']
    centroid = row['geometry'].centroid
    lon, lat = centroid.x, centroid.y

    half_size_m = pixel_size_m * half_size_pixels

    def meters_to_degrees_lon(meters, lat_deg):
        return meters / (111320.0 * max(1e-6, math.cos(math.radians(lat_deg))))
    def meters_to_degrees_lat(meters):
        return meters / 110540.0

    dlon = meters_to_degrees_lon(half_size_m, lat)
    dlat = meters_to_degrees_lat(half_size_m)

    # ✅ Be explicit about projection (CRS) and geodesic flag
    region = ee.Geometry.Rectangle(
        [lon - dlon, lat - dlat, lon + dlon, lat + dlat],
        'EPSG:4326',  # proj
        False         # geodesic
    )

    out_file = os.path.join(img_path, f"{code_val}.tif")
    if os.path.exists(out_file):
        continue

    ic = (ee.ImageCollection(collection)
            .filterBounds(region)
            .filterDate(start_date, end_date))

    # Wrap .getInfo() in try in case EE chokes on a geometry
    try:
        if ic.size().getInfo() == 0:
            print(f"⚠️ No image for {code_val} (empty collection)")
            continue
    except Exception as e:
        print(f"⚠️ Skipping {code_val}: {e}")
        continue

    image = ee.Image(ic.first()).clip(region)

    try:
        geemap.ee_export_image(
            image,
            filename=out_file,
            scale=pixel_size_m,
            region=region,
            crs="EPSG:4326",  # optional but nice to be explicit
            file_per_band=False
        )

    except Exception as e:
        print(f"⚠️ Failed {code_val}: {e}")
        continue


print("✅ All images downloaded.")



