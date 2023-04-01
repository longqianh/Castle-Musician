import io
from PIL import Image

def bytes_to_img(img_bytes):
    
    buf = io.BytesIO(img_bytes)
    img = Image.open(buf)
    return img