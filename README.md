# HSI Land Cover Classification Using Random Forest

This project performs pixel-wise land cover classification on AVIRIS-NG hyperspectral imagery using a Random Forest classifier. It includes false color composite visualization, PCA-RGB rendering, full image prediction, and GeoTIFF export with geospatial metadata.

The goal is to classify urban and natural land cover types from raw hyperspectral data and produce a publication-ready classified map.

## Features

> False Color Composite (NIR-Red-Green) with 2–98 percentile contrast stretch  
> PCA-RGB visualization for quick spectral overview  
> Pixel-wise Random Forest classification with stratified train/test split  
> Full image prediction with custom 9-class color legend  
> GeoTIFF export with CRS and affine transform via rasterio  
> Model persistence using joblib for reuse without retraining  

## Tech Stack

> Python  
> scikit-learn  
> spectral (SPy)  
> rasterio  
> scikit-image  
> matplotlib  
> joblib  

## Project Structure

```
AVIRIS_NG_Ahmedabad.hdr          input HSI image
AVIRIS_NG_Ahmedabad_Classified.hdr   ground truth labels
classify.py                      main pipeline script
rf_model.joblib                  saved Random Forest model
classified_output.tif            GeoTIFF classified output
requirements.txt                 dependencies
README.md
```

## Land Cover Classes

| Class ID | Name        | Color      |
|----------|-------------|------------|
| 1        | Lake        | Dark Cyan  |
| 2        | River       | Blue       |
| 3        | Grass       | Lime Green |
| 4        | Vegetation  | Dark Green |
| 5        | China Mosaic| Yellow     |
| 6        | Tin Shed    | Cyan       |
| 7        | Concrete    | Red        |
| 8        | Asphalt     | Black      |
| 9        | Bare Ground | Orchid     |

## Pipeline

1. **Load Data** — HSI cube and ground truth loaded via `spectral` and validated for shape match
2. **Visualize** — FCC (NIR-R-G bands 64, 44, 25) and PCA-RGB composites displayed
3. **Extract Pixels** — Valid class pixels (1–9) masked and flattened for supervised learning
4. **Train/Test Split** — Stratified 80/20 split, downsampled to 5000 samples for speed
5. **Train Random Forest** — 10-tree RF trained on spectral pixel vectors
6. **Classify Full Image** — All valid pixels predicted and rebuilt into spatial map
7. **Export** — Classified map saved as GeoTIFF with EPSG:4326 CRS and affine transform

## How to Run

```bash
git clone <repo-link>
cd <repo-name>
pip install -r requirements.txt
python classify.py
```

## Requirements

```
numpy
matplotlib
scikit-learn
spectral
rasterio
scikit-image
joblib
```

## Results

The Random Forest classifier predicts one of 9 land cover classes for every valid pixel in the AVIRIS-NG scene. Output is saved as:
- A color-coded matplotlib plot with class legend
- A GeoTIFF (`classified_output.tif`) ready for use in QGIS or ArcGIS

## Known Limitations

- Affine transform uses hardcoded approximate coordinates — replace with actual HDR metadata for accurate geolocation
- `n_estimators=10` is low; increase to 100+ for better accuracy
- No accuracy score is printed after training — add `clf.score(X_test, y_test)` for evaluation
- Training uses only 5000 samples; increase for better generalization on large scenes

## Future Improvements

> Use actual geotransform extracted from HDR metadata  
> Add accuracy report and confusion matrix on test split  
> Increase RF estimators to 100+  
> Compare with SVM or 1D-CNN classifier  
> Deploy via Streamlit or QGIS plugin  
> Support multi-scene batch processing  

## Disclaimer

This project is for research and educational purposes only. Outputs should not be used as authoritative land cover maps without proper field validation.
