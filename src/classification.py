#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import rasterio as rio
from rasterio.enums import Resampling
import json
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from PIL import ImageColor
import skimage as ski
from skimage.exposure import rescale_intensity
import scipy
from rasterio.features import shapes
from dotenv import load_dotenv

load_dotenv()


# ## Data Preparation 
# 
# This is where you supposed to put your file location so that it can be loaded to the script

# In[2]:


lc_dir = '../data/lcb.json'
sample_dir = "../data/lc_bangore.geojson"
sentinel_2 = "../data/Sentinel2_Composite.tif"


# In[3]:


# Load Land Cover Parameter
lc = json.load(open(lc_dir))
lc_df = pd.DataFrame(lc)
lc_df["values_normalize"] = lc_df.index + 1
lc_df["palette"] = "#" + lc_df["palette"]

# Mapping from old to new values
values = lc_df["values"].to_list()
values_norm = lc_df["values_normalize"].to_list()
palette = lc_df["palette"].to_list()
labels = lc_df["label"].to_list()
dict_values = {}
dict_label = {}
dict_palette = {}
dict_palette_hex = {}
for x in range(0, len(values)):
    dict_values[values[x]] = values_norm[x]
    dict_label[values_norm[x]] = labels[x]
    dict_palette[values_norm[x]] = ImageColor.getrgb(palette[x])
    dict_palette_hex[values_norm[x]] = palette[x]

# Create colormap from values and palette
cmap = ListedColormap(palette)

# Patches legend
patches = [
    mpatches.Patch(color=palette[i], label=labels[i]) for i in range(len(values))
]
legend = {
    "handles": patches,
    "bbox_to_anchor": (1.05, 1),
    "loc": 2,
    "borderaxespad": 0.0,
}

lc_df


# In[4]:


sample = gpd.read_file(sample_dir)
sample["value"] = sample["lc"].map(dict_values)
sample["label"] = sample["value"].map(dict_label)

# Plot sample
sample.plot(column="value", cmap=cmap, markersize=1)
plt.legend(**legend)

# Sample with extract
sample_extract = sample.copy()
coords = [
    (x, y) for x, y in zip(sample_extract["geometry"].x, sample_extract["geometry"].y)
]
print(sample_extract.shape)


# ### Showing and Extract Landsat Raster Values ###
# This part is where we can see how the Landsat image look alike.
# 
# You should adjust which band to use and the scale min max value to show your data
# 
# This part also show you how we sample the value of pixel in the Landsat with our sample

# In[6]:


# Load landsat image
sentinel_2 = rio.open(sentinel_2)
sentinel_2_image = sentinel_2.read() /1e4


# In[9]:


# False color composite
out_range = (0, 3)
red = rescale_intensity(sentinel_2_image [3], in_range=(0.01, 0.8), out_range=out_range)
green = rescale_intensity(sentinel_2_image [2], in_range=(0.01, 0.8), out_range=out_range)
blue = rescale_intensity(sentinel_2_image [1], in_range=(0.01, 0.7), out_range=out_range)
arr_image = np.stack(
    [red, green, blue]
).T
composite = np.rot90(np.flip(arr_image, 1), 1)

# Plot landsat image
plt.imshow(composite)

# Extract raster value
sentinel_2_extract = np.stack(
    [x for x in sentinel_2.sample(coords)]
) / 1e4
sample_extract[["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8","B8A","B9", "B11", "B12"]] = sentinel_2_extract
sample_extract.head()


# In[10]:


# Split sample to train and test
seeds = 2
train, test = train_test_split(sample_extract, train_size=0.7, random_state=seeds)
print(f'Train size: {len(train)}\nTest size: {len(test)}')


# In[11]:


# Make random forest model
predictors = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8","B8A","B9", "B11", "B12"]
model = RandomForestClassifier(100)
model.fit(
    train[predictors],
    train["value"]
)


# In[19]:


# Test model
test_apply = model.predict(test[predictors])

# Confusion matrix
cm = confusion_matrix(test['value'], test_apply)
display = ConfusionMatrixDisplay(cm)
display.plot()

# Report
report = classification_report(test['value'], test_apply)
print(report)


# In[22]:


# Load image
combine_image = sentinel_2_image
image_transpose = combine_image.T
transpose_shape = image_transpose.shape
table_image = pd.DataFrame(
    image_transpose.reshape(-1, transpose_shape[2]),
    columns=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8","B8A","B9", "B11", "B12"],
)
table_image


# In[23]:


prediction = model.predict(table_image[predictors])
prediction


# In[32]:


# Prediction to image again
prediction_image = np.rot90(np.flip(prediction.reshape(transpose_shape[0], transpose_shape[1]), 1), 1)

# Show to plot
plt.figure(figsize=(10, 10))
plt.imshow(prediction_image, cmap=cmap, interpolation="nearest")
plt.legend(**legend)


# In[33]:


# Save image to geotiff
output = rio.open(
    "../data/LULC.tif",
    "w",
    "COG",
    count=1,
    width=prediction_image.shape[1],
    height=prediction_image.shape[0],
    crs=sentinel_2.crs,
    transform=sentinel_2.transform,
    dtype="uint8",
    nodata=0,
    compress="lzw",
    resampling="mode",
    tiled=True,
)
output.write_colormap(1, dict_palette)
output.write(prediction_image, 1)
output.close()


# ## Image Segmentation ##
# 
# Sometime the result of land cover classification too many salt and pepper effect
# 
# This section will show you how to create a segmented land cover where this affect is dismissed
# 
# We will generate a segmentation using SLIC algorithms then used it to calculate zonal statistics per segment in the land cover data we have generated

# In[35]:


# Image segmentation

# Do uniform filter to composite image
seed_image = composite
plt.figure(figsize=(10, 10))
plt.imshow(seed_image)

# Segmentation
segment = ski.segmentation.slic(
    seed_image, n_segments=10000, compactness=5, sigma=5
)
plt.figure(figsize=(10, 10))
plt.imshow(ski.segmentation.mark_boundaries(composite, segment, outline_color=(0, 255, 255)))


# ### Calculating Mode of Land Cover Per Segment ###
# 
# After the segment is created, each segment will be used to calculate the mode of land cover that overlayed it
# 
# Then show the result just like the previous non segmented one

# In[36]:


# Get the mode of each segment
segment_unique = np.unique(segment)
lc_segment = segment.copy()
for x in segment_unique:
    lc_segment[segment == x] = scipy.stats.mode(prediction_image[segment == x]).mode
lc_segment


# In[37]:


# Show to plot the segmented LC
plt.figure(figsize=(10, 10))
plt.imshow(lc_segment, cmap=cmap, interpolation="nearest")
plt.legend(**legend)


# In[39]:


# Save image to geotiff
output = rio.open(
    "../data/LULC_Segment.tif",
    "w",
    "COG",
    count=1,
    width=lc_segment.shape[1],
    height=lc_segment.shape[0],
    crs=sentinel_2.crs,
    transform=sentinel_2.transform,
    dtype="uint8",
    nodata=0,
    compress="lzw",
    resampling="mode",
    tiled=True,
)
output.write_colormap(1, dict_palette)
output.write(lc_segment, 1)
output.close()


# In[40]:


# Convert raster to shapefile
lc_vector = [{ "type": "Feature", "properties": { "lc": x[1] }, "geometry": x[0] } for x in shapes(lc_segment.astype('uint8'), transform=sentinel_2.transform)]
lc_vector = json.dumps({
    "type": "FeatureCollection",
	"properties": {},
	"features": lc_vector
})

# Read as geodataframe
lc_df = gpd.read_file(lc_vector, driver='GeoJSON')

# Plot it
lc_df.plot(column="lc", cmap=cmap)
plt.legend(**legend)


# In[45]:


# Add another column such as palette and label
lc_df["palette"] = lc_df["lc"].map(dict_palette_hex)
lc_df["label"] = lc_df["lc"].map(dict_label)

# Save the file
lc_df.to_file("../data/LULC_Shapefile.shp")

