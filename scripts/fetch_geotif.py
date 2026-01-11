import asf_search as asf
import os
from pathlib import Path

# Define AOI (WKT) and search
wkt = "POLYGON((6.6 50.4, 7.0 50.4, 7.0 50.7, 6.6 50.7, 6.6 50.4))"

# First try: broad search to debug
results = asf.geo_search(
    intersectsWith=wkt,
    platform=asf.PLATFORM.SENTINEL1,
    start="2021-07-13",  # Peak flood dates
    end="2021-07-17",    # Just after peak
)

print(f"Broad search found {len(results)} products")

# If we found some, filter for GRD + dual-pol
if len(results) > 0:
    results = [r for r in results if 'GRD' in r.properties.get('processingLevel', '') 
               and 'VV' in r.properties.get('polarization', '') 
               and 'VH' in r.properties.get('polarization', '')]
    print(f"After GRD + VV+VH filter: {len(results)} products")
    if len(results) > 0:
        # Print available properties to debug
        first_result = results[0]
        print(f"First result properties: {list(first_result.properties.keys())}")
        # Use the correct key
        scene_name = first_result.properties.get('sceneName') or first_result.properties.get('granuleName') or first_result.properties.get('fileID')
        print(f"First result: {scene_name}")
else:
    print("No Sentinel-1 products found in this area/date range!")
    print("Check: bbox correct? Dates have S1 coverage?")

# Create output directory
output_dir = Path("/Users/alexwebb/laptop_coding/floodai/InferSAR/data/1raw")
# output_dir.mkdir(parents=True, exist_ok=True)

# Get credentials from env
username = os.getenv("ASF_USERNAME", "nimchimpski")
password = os.getenv("ASF_PASSWORD", "l#7BeJbPC*TKl6")

# Create a session with credentials
session = asf.ASFSession()
session.auth = (username, password)

# Download all results
print(f"Found {len(results)} products. Downloading...")
for i, result in enumerate(results):
    try:
        result.download(path=str(output_dir), session=session)
        scene = result.properties.get('sceneName', 'unknown')
        print(f"[{i+1}/{len(results)}] Downloaded: {scene}")
    except Exception as e:
        scene = result.properties.get('sceneName', 'unknown')
        print(f"Error downloading {scene}: {e}")