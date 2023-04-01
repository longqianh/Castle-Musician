import io
from PIL import Image

def bytes_to_img(img_bytes):
    
    buf = io.BytesIO(img_bytes)
    img = Image.open(buf)
    return img

if __name__ == '__main__':
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.outputs import OutputKeys

    img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_large_en')
    result = img_captioning({'image': 'https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg'})
    print(result[OutputKeys.CAPTION]) # 'a bunch of donuts on a wooden board with popsicle sticks'
    # 目前caption支持了batch inference，方式非常简单，具体如下：
    result = img_captioning([{'image': 'https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg'} for _ in range(3)], batch_size=2)
    for r in result:
        print(r[OutputKeys.CAPTION]) # 'a bunch of donuts on a wooden board with popsicle sticks'