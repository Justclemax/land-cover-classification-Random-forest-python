import geopandas as gpd
import matplotlib.patches as mpatches
from PIL import ImageColor
from loguru import logger
import json
import os
import  numpy as np
import argparse
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import  rasterio as rio
from dotenv import  load_dotenv
from skimage.exposure import rescale_intensity

load_dotenv()

def load_color_data(path: str) -> pd.DataFrame:
    """
    Charge les données de couleur à partir d'un fichier JSON et les retourne sous forme de DataFrame.
    """
    with open(path) as file:
        landcover_color = json.load(file)
    return pd.DataFrame(landcover_color)

def transform_color(path: str):
    """
    Transforms color data by adding standardized columns and creating mappings
    for values, palettes, and labels.
    """
    # Charger les données
    data = load_color_data(path)

    # Normalisation et formatage
    data["values_normalize"] = data.index + 1
    data["palette"] = "#" + data["palette"]

    # Creating lists for processing
    values = data["values"].to_list()
    values_norm = data["values_normalize"].to_list()
    palette = data["palette"].to_list()
    labels = data["label"].to_list()

    # Dictionaries for correspondence
    dict_values = {values[x]: values_norm[x] for x in range(len(values))}
    dict_label = {values_norm[x]: labels[x] for x in range(len(values))}
    dict_palette = {values_norm[x]: ImageColor.getrgb(palette[x]) for x in range(len(values))}
    dict_palette_hex = {values_norm[x]: palette[x] for x in range(len(values))}

    # Colormap
    cmap = ListedColormap(palette)

    # Légende
    patches = [
        mpatches.Patch(color=palette[i], label=labels[i]) for i in range(len(values))
    ]
    legend = {
        "handles": patches,
        "bbox_to_anchor": (1.05, 1),
        "loc": 2,
        "borderaxespad": 0.0,
    }


    return {
        "data": data,
        "colormap": cmap,
        "legend": legend,
        "dict_values": dict_values,  # Corrected key
        "dict_label": dict_label,
        "dict_palette": dict_palette,
        "dict_palette_hex": dict_palette_hex
    }

def process_and_plot_sample(sample_path: str, transform_result: dict) -> object:
    """
    Load a geospatial file, apply value and label transformations,
    and displays the data.
    :rtype: object
    """
    try:
        sample = gpd.read_file(sample_path)

        if "lc" not in sample.columns:
            raise KeyError("La colonne 'lc' est manquante dans les données d'entrée.")

        if not transform_result:
            raise ValueError("Le dictionnaire 'transform_result' est vide.")

        # Mapper les valeurs et les étiquettes
        sample["value"] = sample["lc"].map(transform_result["dict_values"])
        sample["label"] = sample["value"].map(transform_result["dict_label"])

        # Tracer les données
        ax = sample.plot(column="value", cmap=transform_result["colormap"], markersize=1, figsize=(10, 8))
        plt.legend(**transform_result["legend"])
        plt.title("Visualisation des données d'échantillon")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

        # Extraction des coordonnées
        sample_extract = sample.copy()
        coords = [
            (geom.x, geom.y) for geom in sample_extract.geometry if geom.geom_type == "Point"
        ]
        logger.info(f"Nombre de points dans l'échantillon : {sample_extract.shape[0]}")
        logger.info(f"Coordonnées des points extraits (premiers 5 points) : {coords[:5]}")

        return sample_extract, coords

    except KeyError as e:
        logger.error(f"Key error : {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data processing : {e}")
        raise

def load_image_getting_optic_tensor(path_img: str, out_range :tuple =(0,1)):
    try:
        with open(path_img) as file :
            optic_img = rio.open(file).read()
        red = rescale_intensity(optic_img[4],in_range =(0.1, 0.4), out_range= out_range)
        blue = rescale_intensity(optic_img[5],in_range=(0.05, 0.3), out_range= out_range)
        green = rescale_intensity(optic_img[6],in_range=(0.025, 0.25), out_range= out_range)
        arr_image = np.stack(
            [red, green, blue]
        ).T

        composite = np.rot90(np.flip(arr_image, axis=1), k=1)
        plt.show(composite)


        return  optic_img * 1e4

    except Exception as e :
        logger.critical(f" {e}")
    return None

def extraction_data_in_optic_image(path) -> pd.DataFrame:
    """
    Extract raster values from the optical image and return a DataFrame with the data.
    """
    try:
        sample_extract, coords = process_and_plot_sample(sample_path=path, transform_result={})

        # Extract the corresponding raster values for the coordinates
        optic_img = load_image_getting_optic_tensor(path)

        if optic_img is None:
            raise ValueError("Error loading optical image data.")
        #image_extract = np.stack(
           # [x for x in optic_img.sample(coords)]
       # ) / 1e4
        image_extract = np.stack([optic_img[:, float(coord[1]), float(coord[0])] for coord in coords]) / 1e4

        sample_extract[["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9" ,"B11", "B12"]] = image_extract
        return sample_extract

    except Exception as e:
        logger.error(f"Error extracting data from optical image: {e}")
        raise


def main():
    # Initialisation
    parser = argparse.ArgumentParser(description="Process and visualize land cover data.")
    parser.add_argument("--color_data", required=True, help="Path to the land cover color JSON file.")
    parser.add_argument("--sample_data", required=True, help="Path to the GeoJSON or Shapefile sample data.")
    parser.add_argument("data_path_raster_image", help= "Path to raster image", default=os.getenv("PATH_DATA_IMAGE"))


    # Parse les arguments
    args = parser.parse_args()

    try:
        # Load and transform color data
        logger.info("Loading and transforming colour data...")
        transform_result = transform_color(args.color_data)

        # Load, transform and plot sample data
        logger.info("Processing and displaying sample data...")
        process_and_plot_sample(args.sample_data, transform_result)

    except Exception as e:
        logger.critical(f"Critical error in the main script : {e}")


if __name__ == "__main__":
    main()