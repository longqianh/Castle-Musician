import io
from PIL import Image


def bytes_to_img(img_bytes):
    buf = io.BytesIO(img_bytes)
    img = Image.open(buf)
    return img


if __name__ == "__main__":
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.outputs import OutputKeys
  
    img_captioning = pipeline(
        Tasks.image_captioning, model="damo/ofa_image-caption_coco_large_en",
    )
    result = img_captioning(
        {
            "image": "./assets/icon.png"
        }
    )
    print(
        result[OutputKeys.CAPTION]
    )  # 'a bunch of donuts on a wooden board with popsicle sticks'
