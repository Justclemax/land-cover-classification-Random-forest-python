import geopandas as gpd
import matplotlib.patches as mpatches
from PIL import ImageColor
from loguru import logger
import json

import  numpy as np

import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import  rasterio as rio

from skimage.exposure import rescale_intensity



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

        plt.pause(10)
        plt.close()

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

def load_image_getting_optic_tensor(path_img: str, out_range =(0,3)):
    try:

        optic_img = rio.open(path_img).read() / 1e4
        red = rescale_intensity(optic_img[3],in_range =(0.1, 0.8), out_range= out_range)
        blue = rescale_intensity(optic_img[2],in_range=(0.01, 0.8), out_range= out_range)
        green = rescale_intensity(optic_img[1],in_range=(0.01, 0.7), out_range= out_range)
        arr_image = np.stack(
            [red, green, blue]
        ).T

        composite = np.rot90(np.flip(arr_image, axis=1), k=1)
        plt.title('Raster image')
        plt.imshow(composite)

        #plt.show()
        plt.pause(10)
        plt.close()
        return  optic_img

    except Exception as e :
        logger.critical(f" {e}")
        raise

def extraction_data_in_optic_image(sample_path, path_img, transform_result: dict) -> pd.DataFrame:
    """
    Extract raster values from the optical image and return a DataFrame with the data.
    """
    try:
        sample_extract, coords = process_and_plot_sample(sample_path=sample_path, transform_result=transform_result)
        # Extract the corresponding raster values for the coordinates
        optic_img = load_image_getting_optic_tensor(path_img)

        if optic_img is None:
            raise ValueError("Error loading optical image data.")
        #image_extract = np.stack(
           # [x for x in optic_img.sample(coords)]
       # ) / 1e4
        image_extract = np.stack([optic_img[:, int(coord[1]), int(coord[0])] for coord in coords]) / 1e4

        sample_extract[["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9" ,"B11", "B12"]] = image_extract

        return sample_extract

    except Exception as e:
        logger.error(f"Error extracting data from optical image: {e}")
        raise


