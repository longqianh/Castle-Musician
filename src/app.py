import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

torch.backends.cudnn.benchmark = False
import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from PIL import Image
from pathlib import Path
from pydub import AudioSegment
from pathlib import Path
import time
from audio_utils import sound_to_wav, reduce_noise, get_wav_audio
from image_utils import bytes_to_img
from model_utils import init_model


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from model_utils import AudioLdmWrapper


dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
# ===== image model 1 =====
# img2text_model_name="nlpconnect/vit-gpt2-image-captioning"
# img2text_model_backend='huggingface'
# ===== image model 2 =====
img2text_model_name = "damo/ofa_image-caption_coco_large_en"
img2text_model_backend = "modelscope"
# ===== audio model =====
audio_model_name = "audioldm-s-full-v2"
audio_model_ckpt = f"{dirname}/assets/audioldm-full-s-v2.ckpt"
save_path = f"{dirname}/output/"

os.makedirs(save_path, exist_ok=True)
img = Image.open(f"{dirname}/assets/icon.png")


st.set_page_config(
    page_title="Castle Musician",
    page_icon=img,
    layout="wide",
)
st.title("Castle Musician")
st.write(
    """
Music is a powerful form of art that can touch our emotions and thoughts
 through elements such as melody, rhythm, and harmony. Castle musician is
   an AIGC project designed for high-quality music creation. The ultimate 
   goal is to use AIGC-created music on neurological patients with fMRI or fNIRS signal monitoring.
"""
)

sample_rate = 16000  # 44100 samples per second
time_mask_ratio_start_and_end = (0.10, 0.15)
freq_mask_ratio_start_and_end = (1.0, 1.0)

# image2text_model, audio_tool = init_model(
#     img2text_model_name, img2text_model_backend, audio_model_name, audio_model_ckpt
# )

img_captioning = pipeline(
    Tasks.image_captioning,
    model=img2text_model_name,
    # device=device
)
audio_tool = AudioLdmWrapper(audio_model_name, audio_model_ckpt)

musician_data = st.selectbox(
    "Data type to musician",
    ("Image", "Text", "fMRI (building...)", "fNIRS (building...)"),
)


guidance_scale = 4.5
n_candidate = 3  # Generate n_candidate_gen_per_text times and select the best
# 'musician instrument',

transfer_strength = 0.55

global_is_ready = False

if musician_data == "Image":
    with st.container():
        duration_mode = st.select_slider(
            "Select music duration", options=["short", "medium", "long"]
        )
        duration_mode_dict = {"short": 10.0, "medium": 20.0, "long": 45.0}
        duration = duration_mode_dict[duration_mode]

        ddim_steps_mode = st.select_slider(
            "Select inference mode", options=["fast", "medium", "slow"]
        )
        step_mode_dict = {"fast": 100, "medium": 200, "slow": 400}
        ddim_steps = step_mode_dict[ddim_steps_mode]
        uploaded_img = st.file_uploader(
            "Image to musician", type=["jpeg", "jpg", "png", "bmp"]
        )

        if uploaded_img is not None:
            img_bytes = uploaded_img.getvalue()

            # ===== image model 1 =====
            # res = image2text_model(bytes_to_img(img_bytes))[0]
            # img_caption = res['generated_text']  # nlpconnect/vit-gpt2-image-captioning

            # ===== image model 2 =====
            res = image2text_model(bytes_to_img(img_bytes))
            img_caption = res["caption"][0]  # damo/ofa_image-caption_coco_large_en

            st.image(img_bytes, caption=img_caption, width=400)
            print(type(res), list(res.keys()))
            print("image text =", img_caption)

            # with st.container():
            #     duration=st.slider('Duration',min_value=20,max_value=200)
            #     ddim_steps=st.slider('Computing Steps',min_value=50,max_value=600)
            #     st.write(duration)
            #     st.write(ddim_steps)

            if st.button("Confirm"):
                global_is_ready = True

            if global_is_ready:
                global_is_ready = False
                st.write("Creating music...")
                audio_tool.is_ready_ = True

                # ==== origin style ====
                torch.cuda.empty_cache()
                audio_imagined = audio_tool.text_to_audio(
                    img_caption,
                    None,
                    duration=duration,
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    n_candidate_gen_per_text=n_candidate,
                    save_path=save_path,
                )
                if not isinstance(audio_imagined, bool):
                    file_0 = get_wav_audio(save_path + "audio_0.wav")
                    st.audio(file_0, format="audio/wav")  # sample_rate=sample_rate
                    # st.download_button(label="Download",data=file_0,file_name='music.wav',mime='audio/wav')
                else:
                    st.write("Waiting server...")
                # st.write('Half done.')
                # ==== convert style ====
                time.sleep(1.0)
                print("here in Confirm")
                torch.cuda.empty_cache()
                # audio_imagined_styled1=audio_tool.audio_style_transfer('classic violin',file_path=save_path+'audio_0.wav',\
                #                       transfer_strength=transfer_strength,duration=duration, guidance_scale=guidance_scale,\
                #                       ddim_steps=ddim_steps,save_path=save_path)

                # if not isinstance(audio_imagined_styled1,bool):
                #     reduce_noise(save_path+'audio_styled_0.wav', save_path+'audio_styled_0_rn.wav')
                #     file_0 = get_wav_audio(save_path+'audio_styled_0_rn.wav')
                #     st.audio(file_0, format='audio/wav') # sample_rate=sample_rate
                #     # st.download_button(label="Download",data=file_1,file_name='music.wav',mime='audio/wav')
                # else:
                #     st.write('Waiting server...')

                audio_imagined_styled2 = audio_tool.audio_style_transfer(
                    "piano music",
                    file_path=save_path + "audio_0.wav",
                    transfer_strength=transfer_strength,
                    duration=duration,
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    save_path=save_path,
                )

                if not isinstance(audio_imagined_styled2, bool):
                    reduce_noise(
                        save_path + "audio_styled_0.wav",
                        save_path + "audio_styled_0_rn.wav",
                    )
                    file_1 = get_wav_audio(save_path + "audio_styled_0_rn.wav")
                    st.audio(file_1, format="audio/wav")  # sample_rate=sample_rate
                    # st.download_button(label="Download",data=file_1,file_name='music.wav',mime='audio/wav')
                else:
                    st.write("Waiting server...")

                # if st.button('HighQ'):
                #     audio_imagined_styled_highq=audio_tool.audio_super_resolution(save_path+'audio_styled_0.wav', duration,
                #                 guidance_scale, ddim_steps, n_candidate, time_mask_ratio_start_and_end,\
                #                     freq_mask_ratio_start_and_end, save_path=save_path)
                #     if not isinstance(audio_imagined_styled_highq,bool):
                #         st.audio(audio_imagined_styled_highq[0], format='audio/wav',sample_rate=sample_rate) # sample_rate=sample_rate
                #         st.download_button(label="Download",data=audio_imagined_styled_highq[0],file_name='music_highq_0.wav',mime='audio/wav')
                #     else:
                #         st.write('Waiting server...')

                global_is_ready = True


elif musician_data == "Text":
    with st.container():
        duration_mode = st.select_slider(
            "Select music duration", options=["short", "medium", "long"]
        )
        duration_mode_dict = {"short": 10.0, "medium": 20.0, "long": 45.0}
        duration = duration_mode_dict[duration_mode]

        ddim_steps_mode = st.select_slider(
            "Select inference mode", options=["fast", "medium", "slow"]
        )
        step_mode_dict = {"fast": 100, "medium": 200, "slow": 400}
        ddim_steps = step_mode_dict[ddim_steps_mode]
        text = st.text_input("Text to musician", value="castle in the sky")
        if st.button("Confirm"):
            global_is_ready = True

        if global_is_ready:
            global_is_ready = False
            # audio_tool.is_ready_=True
            st.write("Creating music...")

            # with st.container():
            #     duration=st.slider('Duration',min_value=5,max_value=40)
            #     ddim_steps=st.slider('Computing Steps',min_value=50,max_value=1000)
            #     st.write(duration)
            #     st.write(ddim_steps)

            # ==== origin style ====
            save_path = save_path
            torch.cuda.empty_cache()
            print("here in Text, step1")
            audio_imagined = audio_tool.text_to_audio(
                text,
                None,
                duration=duration,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                n_candidate_gen_per_text=n_candidate,
                save_path=save_path,
            )
            if not isinstance(audio_imagined, bool):
                file_0 = get_wav_audio(save_path + "audio_0.wav")
                st.audio(file_0, format="audio/wav")  # sample_rate=sample_rate
                # st.download_button(label="Download",data=file_0,file_name='music.wav',mime='audio/wav')
            else:
                st.write("Waiting server...")

            # ==== convert style ====
            time.sleep(1.0)
            print("here in Text, step2")
            torch.cuda.empty_cache()
            audio_imagined_styled = audio_tool.audio_style_transfer(
                "piano music",
                file_path=save_path + "audio_0.wav",
                transfer_strength=transfer_strength,
                duration=duration,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                save_path=save_path,
            )

            if not isinstance(audio_imagined_styled, bool):
                reduce_noise(
                    save_path + "audio_styled_0.wav",
                    save_path + "audio_styled_0_rn.wav",
                )
                file_1 = get_wav_audio(save_path + "audio_styled_0_rn.wav")
                st.audio(file_1, format="audio/wav")  # sample_rate=sample_rate
                # st.download_button(label="Download",data=file_1,file_name='music.wav',mime='audio/wav')
            else:
                st.write("Waiting server...")

            # if st.button('HighQ'):
            #     if not isinstance(audio_imagined_styled_highq,bool):

            #         audio_imagined_styled_highq=audio_tool.audio_super_resolution(save_path+'audio_styled_0.wav', duration,
            #                     guidance_scale, ddim_steps, n_candidate, time_mask_ratio_start_and_end,\
            #                             freq_mask_ratio_start_and_end, save_path=save_path)

            #         file_1 = get_wav_audio(save_path+'audio_styled_0.wav')
            #         st.audio(file_1, format='audio/wav',sample_rate=sample_rate)
            #         st.download_button(label="Download",data=file_1,file_name='music_highq_0.wav',mime='audio/wav')
            #     else:
            #         st.write('Waiting server...')

            global_is_ready = True


elif musician_data == "fMRI (building...)":
    with st.container():
        music_path_fmri = Path(f"{dirname}/assets/castle_in_the_sky.mp3")
        audio = AudioSegment.from_file(music_path_fmri)
        audio_wav_fmri = io.BytesIO()
        audio.export(audio_wav_fmri, format="wav")
        st.audio(audio_wav_fmri, format="audio/wav")

elif musician_data == "fNIRS (building...)":
    with st.container():
        music_path_fnirs = Path(f"{dirname}/assets/Summer.mp3")
        audio = AudioSegment.from_file(music_path_fnirs)
        audio_wav_fnirs = io.BytesIO()
        audio.export(audio_wav_fnirs, format="wav")
        st.audio(audio_wav_fnirs, format="audio/wav")
