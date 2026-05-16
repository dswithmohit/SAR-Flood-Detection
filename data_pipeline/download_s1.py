import os
import ee
import geemap

def initialize_gee():
    """Initializes and authenticates Google Earth Engine with a Cloud Project."""
    # Updated with your project ID
    project_id = 'ee-mohit-flood' 
    
    try:
        ee.Initialize(project=project_id)
        print("GEE Initialized successfully.")
    except Exception as e:
        print("GEE initialization failed with project ID. Trying re-authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

def get_sar_scene(aoi, start_date, end_date):
    """
    Fetches a Sentinel-1 SAR image collection filtered by AOI and dates.
    Returns a single median composite image in VV polarization.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        # IW (Interferometric Wide) is the standard mode for land
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        # We look for VV polarization (best for water vs land contrast)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        # Descending orbit passes over Bangladesh in the early morning
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    )
    
    # Take the median value of the pixels to clear out random noise/ships
    return collection.select(["VV"]).median()

def main():
    initialize_gee()
    
    # Create local data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Bounding Box for Sylhet, Bangladesh [Min Long, Min Lat, Max Long, Max Lat]
    sylhet_bbox = [91.3, 24.5, 92.2, 25.3]
    aoi = ee.Geometry.Rectangle(sylhet_bbox)
    
    # 1. Define time windows
    pre_flood_start, pre_flood_end = "2024-05-01", "2024-05-20"   # Dry/Normal season
    post_flood_start, post_flood_end = "2024-08-20", "2024-08-30" # Peak 2024 Flood
    
    print("Fetching SAR data from GEE...")
    pre_flood_img = get_sar_scene(aoi, pre_flood_start, pre_flood_end)
    post_flood_img = get_sar_scene(aoi, post_flood_start, post_flood_end)
    
    # 2. Export options (setting scale to 30 meters to keep file sizes manageable)
    # Sentinel-1 natively goes to 10m, but 30m keeps things fast for a 3-day MVP.
    export_params = {
        "scale": 40,
        "region": aoi,
        "file_per_band": False 
    }
    
    print("Downloading Pre-Flood GeoTIFF...")
    geemap.ee_export_image(
        pre_flood_img, 
        filename="data/raw/pre_flood.tif", 
        **export_params
    )
    
    print("Downloading Post-Flood GeoTIFF...")
    geemap.ee_export_image(
        post_flood_img, 
        filename="data/raw/post_flood.tif", 
        **export_params
    )
    
    print("Downloads complete! Check your data/raw/ folder.")

if __name__ == "__main__":
    main()