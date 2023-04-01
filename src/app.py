import streamlit as st
import pandas as pd
import numpy as np
import io 
from PIL import Image

from pathlib import Path
from pydub import AudioSegment
from pathlib import Path
from audio_utils import sound_to_wav
from image_utils import bytes_to_img
from model_utils import init_img2text_model, init_text2audio_model,\
      init_audio_refine_model
# ckpt_path='./ckpt/ldm_trimmed.ckpt'


img=Image.open('./assets/icon.png')
st.set_page_config(
    page_title='Castle Musician',
    page_icon=img,
    layout="wide",
)
st.title('Castle Musician')
st.write('''
Music is a powerful form of art that can touch our emotions and thoughts
 through elements such as melody, rhythm, and harmony. Castle musician is
   an AIGC project designed for high-quality music creation. The ultimate 
   goal is to use AIGC-created music on neurological patients with fMRI or fNIRS signal monitoring.
''')

music_path=Path('./assets/castle_in_the_sky.mp3')
audio=AudioSegment.from_file(music_path)
audio_wav = io.BytesIO()
audio.export(audio_wav, format="wav")
sample_rate = 44100  # 44100 samples per second
audio_wavs=[audio_wav,audio_wav,audio_wav]
# st.audio(note_la, sample_rate=sample_rate)
img2text_model_path="nlpconnect/vit-gpt2-image-captioning"
# img2text_model_path='damo/ofa_image-caption_coco_distilled_en'
img2text_model=init_img2text_model(img2text_model_path,model_backend='huggingface')

musician_data = st.selectbox(
    'Data type to musician',
    ('Image', 'Text', 'fMRI (building...)','fNIRS (building...)'))

# 'musician instrument',

if musician_data=='Image':
        
    with st.container():
        uploaded_img=st.file_uploader("Image to musician", type=["jpeg","jpg", "png", "bmp"])
        
        if uploaded_img is not None:
            img_bytes = uploaded_img.getvalue() 
            img_caption=img2text_model(bytes_to_img(img_bytes))[0]['generated_text']
            st.image(img_bytes, caption=img_caption, use_column_width=True) # todo: resize to smaller

        for audio in audio_wavs:
            st.audio(audio_wav, format='audio/wav') # sample_rate=sample_rate

elif musician_data=="Text":
    text=st.text_input('Text to musician', value='castle in the sky')
    st.audio(audio_wavs[0], format='audio/wav') # sample_rate=sample_rate
    pass
elif musician_data=="fMRI":
    pass
elif musician_data=="fNIRS":
    pass