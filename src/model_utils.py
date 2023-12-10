import streamlit as st
import torch
import yaml

# from audioldm import LatentDiffusion, seed_everything
# from audioldm.utils import default_audioldm_config, get_duration,\
#   get_bit_depth, get_metadata, download_checkpoint, save_wave
# from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
# from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm import (
    text_to_audio,
    style_transfer,
    build_model,
    save_wave,
    get_time,
    # round_up_duration,
    # get_duration,
    super_resolution_and_inpainting,
)


class AudioLdmWrapper:
    def __init__(self, model_name, ckpt_path, random_seed=0):
        self.random_seed = random_seed
        self.is_ready_ = False
        self.model = build_model(ckpt_path=ckpt_path,model_name=model_name)
        self.is_ready_ = True
        print("[GZY]: AudioLDM model init done")

    def text_to_audio(
        self,
        text,
        file_path,
        duration,
        guidance_scale,
        ddim_steps,
        n_candidate_gen_per_text,
        save_path=None,
    ):
        if not self.is_ready_:
            return False
        self.is_ready_ = False
        waveform = text_to_audio(
            self.model,
            text,
            file_path,
            self.random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            batchsize=1,
        )

        if save_path is not None:
            save_wave(waveform, save_path, name="audio")
        self.is_ready_ = True
        return waveform

    def audio_style_transfer(
        self,
        text,
        file_path,
        transfer_strength,
        duration,
        guidance_scale,
        ddim_steps,
        save_path=None,
    ):
        if not self.is_ready_:
            return False
        self.is_ready_ = False
        waveform = style_transfer(
            self.model,
            text,
            file_path,
            transfer_strength,
            self.random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            batchsize=1,
        )
        waveform = waveform[:, None, :]
        if save_path is not None:
            save_wave(waveform, save_path, name="audio_styled")
        self.is_ready_ = True
        return waveform

    def audio_super_resolution(
        self,
        text,
        file_path,
        duration,
        guidance_scale,
        ddim_steps,
        n_candidate_gen_per_text,
        time_mask_ratio_start_and_end,
        freq_mask_ratio_start_and_end,
        save_path=None,
    ):
        if not self.is_ready_:
            return False
        self.is_ready_ = False
        waveform = super_resolution_and_inpainting(
            self.model,
            text=text,
            original_audio_file_path=file_path,
            seed=self.random_seed,
            ddim_steps=ddim_steps,
            duration=duration,
            batchsize=1,
            guidance_scale=guidance_scale,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            time_mask_ratio_start_and_end=time_mask_ratio_start_and_end,  # regenerate the 10% to 15% of the time steps in the spectrogram
            # time_mask_ratio_start_and_end=(1.0, 1.0), # no inpainting
            # freq_mask_ratio_start_and_end=(0.75, 1.0), # regenerate the higher 75% to 100% mel bins
            freq_mask_ratio_start_and_end=freq_mask_ratio_start_and_end,  # no super-resolution
            config=None,
        )
        if save_path is not None:
            save_wave(waveform, save_path, name="audio_highq")
        self.is_ready_ = True
        return waveform


# @st.cache_resource
def init_img2text_model(img2text_model_name, model_backend):
    if model_backend == "huggingface":
        from transformers import pipeline

        img2text_model = pipeline("image-to-text", model=img2text_model_name)

    elif model_backend == "modelscope":
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        from modelscope.outputs import OutputKeys
        img2text_model = pipeline(Tasks.image_captioning, model=img2text_model_name)

    return img2text_model


@st.cache_resource
def init_model(
    img_model_name="nlpconnect/vit-gpt2-image-captioning",
    img_model_end="huggingface",
    audio_model_name="audioldm-s-full-v2",
    audio_model_ckpt="./assets/audioldm-full-s-v2.ckpt",
):
    image_tool = init_img2text_model(img_model_name, img_model_end)
    audio_tool = AudioLdmWrapper(audio_model_name, audio_model_ckpt)
    return image_tool, audio_tool


def predict_audio_from_image(img, model):
    pass


def predict_audio_from_text(
    text,
    model,
    seed=42,
    duration=10,
    ddim_steps=200,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
):
    from audioldm.pipeline import text_to_audio

    return text_to_audio(
        model,
        text,
        duration=duration,
        seed=seed,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=n_candidate_gen_per_text,
    )


def predict_text_from_image(img, model):
    img_caption = model(img)

    return img_caption.detach().cpu().numpy()


if __name__ == "__main__":
    save_path = "./output/"
    audio_model_name = "audioldm-l-full" #"audioldm-s-full-v2"
    # audio_model_ckpt = f"./assets/audioldm-full-s-v2.ckpt"
    audioldm = build_model(model_name=audio_model_name)

    audio = predict_audio_from_text(
        "test",
        audioldm,
        duration=1,
        ddim_steps=20,
        guidance_scale=1,
        n_candidate_gen_per_text=1,
    )
    srate = 16000
    save_wave(audio, save_path, samplerate=srate, name="test")
