# **Land Cover Classification and Segmentation**

This project provides a complete method to classify and segment satellite images, allowing the generation of Land Use/Land Cover (LULC) maps from Sentinel-2 data. The pipeline relies on machine learning techniques (Random Forest) and incorporates advanced tools to reduce noise effects using segmentation.

---

## **Data Sources**
The raster and GeoJSON data used in this project are generated using a Google Earth Engine (GEE) script, available here:  
[Google Earth Engine Script](https://code.earthengine.google.co.in/?scriptPath=users%2Fclementkafwimbi22%2Fdefault%3Araster_and_geojson)

---

### **Setting Up a Google Earth Engine Account**
To use Google Earth Engine and run the script:
1. **Create a Google Earth Engine account:**  
   Go to [Google Earth Engine](https://earthengine.google.com/) and follow the instructions to sign up.
   
2. **Access the GEE Editor:**  
   After your account is approved, sign in and open the online editor. You can paste the script mentioned above to generate your own data.

3. **Export the Data:**  
   - Set up export parameters to save the raster data (GeoTIFF) and vector data (GeoJSON).
   - Download the generated files and place them in the project's `data/` folder.

---

## **Prerequisites**
- Python version **≥ 3.9**.
- Required libraries (installed via `requirements.txt`).

---

## **Project Organization**
- **`data/` :** Contains source data (raster, GeoJSON) and results (classified images and shapefiles).
- **`notebook/classification.ipynb` :** Python script for classification and segmentation.
- **`README.md` :** Detailed project documentation.
- **`src/` :** Reusable Python scripts for analysis and processing.

---

## **Getting Started**
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
Inspired by  https://github.com/ramiqcom from their project  https://github.com/ramiqcom/lc-classification-pythong
#### Sentinel-2 Raster Image
![Original Sentinel-2 Image](https://github.com/user-attachments/assets/fc8d9d8c-37fc-4448-b499-85ad83f305e7)

#### Classification Result
![Classification Output](https://github.com/user-attachments/assets/77a27895-9130-4a8c-918e-d00030142bdf)