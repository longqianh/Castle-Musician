import io
from PIL import Image
import torch

def bytes_to_img(img_bytes):
    buf = io.BytesIO(img_bytes)
    img = Image.open(buf)
    return img


if __name__ == "__main__":
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.outputs import OutputKeys
  
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    img_captioning = pipeline(
        Tasks.image_captioning, model="damo/ofa_image-caption_coco_large_en",
    ).to(device)

    result = img_captioning(
        {
            "image": "./assets/icon.png"
        }
    )
    print(
        result[OutputKeys.CAPTION]
    )  # 'a bunch of donuts on a wooden board with popsicle sticks'
