# Showcase Route TMC Extraction

Generate an ordered TMC list for the showcase route without touching the training pipeline.

## Prerequisites
- Python 3.10+ with pandas/numpy (already in `TrainingModel/requirements.txt`).
- Optional but recommended: `shapely` for fast spatial lookup (`pip install shapely`). The script falls back to a numpy search if shapely is missing.

## Inputs
- TMC identification file (CSV/TSV): e.g. `data/Govt_Data/ShowCase_Data_Set/TMC_Identification.csv`.
- One route polyline in any of these forms:
  - GeoJSON `LineString`: `data/showcase_route_polyline.geojson`.
  - Encoded polyline string in a text file: `data/showcase_route_polyline.txt`.
  - Array of `[lat, lon]` points in JSON: `data/showcase_route_points.json`.

## Run the extractor
From the repository root:
```bash
python3 TrainingModel/scripts/build_showcase_route_tmcs.py \
  --tmc-id-path TrainingModel/data/Govt_Data/ShowCase_Data_Set/TMC_Identification.csv \
  --route-path TrainingModel/data/showcase_route_polyline.geojson \
  --out-path TrainingModel/data/Govt_Data/tmcs_showcase_route.csv \
  --threshold-m 150 \
  --sample-m 50 \
  --expected-directions "S,E" \
  --expected-roads "GA-141,I-285,I-85,I-75"
```

Notes:
- `--expected-directions` / `--expected-roads` are optional filters; leave them off if you are unsure.
- The script prints only safe summaries (counts, IDs) and writes the ordered TMC list to the output CSV.
- If the matched count is tiny (<10) or the total miles are outside 15–80, the script will warn so you can adjust the threshold or verify the polyline.

## How to export the route polyline from the frontend (nextroute)
1) Load your route in the nextroute UI as usual.  
2) Open the browser devtools console. After the route renders, capture the geometry:
   - Locate the route GeoJSON/Polyline in the network response (directions request) or wherever the route object is assembled.
   - Copy the `LineString` coordinates into a new file `data/showcase_route_polyline.geojson` with structure:
     ```json
     { "type": "LineString", "coordinates": [[lon, lat], [lon, lat], ...] }
     ```
   - If the frontend gives an encoded polyline string, paste it into `data/showcase_route_polyline.txt` instead.
3) Do not include any restricted datasets when saving; only the public route geometry is needed.
4) Run the extractor command above to produce `tmcs_showcase_route.csv`.

That CSV is the only artifact required for the showcase route; no training or model code needs to run for this step.
