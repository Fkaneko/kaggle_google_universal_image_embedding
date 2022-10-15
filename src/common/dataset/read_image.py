import logging

from PIL import Image

BACKGROUND_COLOR = (255, 255, 255)


logger = logging.getLogger(__name__)


def convert_to_rgb(image: Image) -> Image:
    # color model: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    if (image.getbands()[-1] in ["A", "a"]) | (image.mode == "P"):
        image = image.convert("RGBA")

    if image.mode == "RGB":
        return image
    elif image.mode == "RGBA":
        # from : http://stackoverflow.com/a/9459208/284318
        image.load()  # needed for split()
        new_rgb_image = Image.new("RGB", image.size, BACKGROUND_COLOR)
        new_rgb_image.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return new_rgb_image
    else:
        return image.convert("RGB")
