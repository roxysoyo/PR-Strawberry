import os
from PIL import Image

image_folderpath = "PR1-Strawberry/Images"


for filename in os.listdir(image_folderpath):
    image_path = os.path.join(image_folderpath, filename)

    image_name = os.path.basename(image_path)
    image_name = image_name.replace(".PNG", "")

    # print(image_name)

    image_path = image_path.replace("\\", "/")
    image_path = image_path.replace(".PNG", ".png")
    print(image_path)
    # Use Pillow to open image files
    image = Image.open(image_path)
    # Save the new three-channel image
    image = image.convert('RGB')
    # Resize image to 96x96 pixels
    new_size = (96, 96)
    resized_image = image.resize(new_size)
    print(type(resized_image))
    # Save the resized image
    resized_image.save("{}.png".format(image_name), format="PNG")
