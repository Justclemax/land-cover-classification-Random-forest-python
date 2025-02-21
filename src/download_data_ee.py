import ee
import os
from dotenv import load_dotenv
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from pathlib import Path
from loguru import logger


CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50



def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.

    return img.addBands(is_cld_shdw)
def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


###################################################################################

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))



def download_ee(AOI, start_date, end_date, raster_name, output_dir='Download_EE', scale=10) -> bool:
    """Download Sentinel-2 composite from Google Earth Engine."""
    # Get Sentinel-2 cloud-masked collection
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, start_date, end_date)

    # Create median composite after applying the cloud and shadow mask
    s2_sr_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
                    .map(apply_cld_shdw_mask)
                    .median())

    # Define the export task
    task = ee.batch.Export.image.toDrive(
        image=s2_sr_median,
        description=raster_name,
        folder=output_dir,
        fileNamePrefix=raster_name,
        scale=scale,
        region=AOI,  # Use the provided AOI
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True}  # Cloud optimized format
    )

    logger.info(f"Starting export task for {raster_name}")
    task.start()

    logger.info(f"Export status for {raster_name}:")

    return task


def read_file(file_path: Path) -> gpd.GeoDataFrame:
    """
    Reads a geospatial file and returns a GeoDataFrame.

    :param file_path: Path to the geospatial file
    :return: GeoDataFrame containing the data
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} doesn't exist")

    logger.info(f"Reading file {file_path}")
    dataset = gpd.read_file(file_path)

    return dataset








if __name__ == '__main__':
    load_dotenv()
    ee.Initialize(project='ai-modelization')

    AOI = ee.Geometry.Rectangle(read_file(Path('../data/bangalore.geojson')).total_bounds.tolist())

    START_DATE = '2020-06-01'
    END_DATE = '2020-09-30'

    download_ee(AOI, START_DATE, END_DATE, 'sentinel_2_SR_composite')