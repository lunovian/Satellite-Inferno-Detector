import pystac_client
import planetary_computer
import requests
from PIL import Image as PILImage
from io import BytesIO
import matplotlib.pyplot as plt

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

time_range = "2020-12-01/2020-12-31"
bbox = [-122.2751, 47.5469, -121.9613, 47.7458]

search = catalog.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=time_range)
items = search.item_collection()
selected_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])

# Print some information about the selected item
print(f"Selected image date: {selected_item.properties['datetime']}")
print(f"Cloud cover: {selected_item.properties['eo:cloud_cover']}%")

# Display available assets
print("\nAvailable assets:")
for asset_key, asset in selected_item.assets.items():
    print(f"- {asset_key}: {asset.title or 'N/A'}")

# Download and display the image
preview_url = selected_item.assets["rendered_preview"].href
response = requests.get(preview_url)
img = PILImage.open(BytesIO(response.content))

# Display with matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title(f"Landsat image from {selected_item.properties['datetime']}")
plt.show()
