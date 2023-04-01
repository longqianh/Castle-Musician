import streamlit as st
import pandas as pd
import numpy as np
import io 
from pathlib import Path
from pydub import AudioSegment
from pathlib import Path
from audio_utils import sound_to_wav
from image_utils import bytes_to_img
from model_utils import init_img2text_model, init_text2audio_model,\
      init_audio_refine_model
# ckpt_path='./ckpt/ldm_trimmed.ckpt'


st.title('Castle Musician')
music_path=Path('../data/castle_in_the_sky.mp3')
audio=AudioSegment.from_file(music_path)
audio_wav = io.BytesIO()
audio.export(audio_wav, format="wav")
sample_rate = 44100  # 44100 samples per second

# st.audio(note_la, sample_rate=sample_rate)
img2text_model_path="nlpconnect/vit-gpt2-image-captioning"
img2text_model=init_img2text_model(img2text_model_path)

with st.container():
    uploaded_img=st.file_uploader("Choose an image", type=["jpeg","jpg", "png", "bmp"])
    
    if uploaded_img is not None:
        img_bytes = uploaded_img.getvalue() 
        img_caption=img2text_model(bytes_to_img(img_bytes))[0]['generated_text']
        st.image(img_bytes, caption=img_caption, use_column_width=True) # todo: resize to smaller
    
    st.audio(audio_wav, format='audio/wav') # sample_rate=sample_rate
    # st.write(f"{img_caption}".format(img_caption))
    # audios = predict_audio(img, model)
    # for audio in audios