import os

os.environ["CUDA_LAUNCH_BLOCKING"] = ""

import streamlit as st
import torch
import yaml
from scipy.io import wavfile

# from audioldm import LatentDiffusion, seed_everything
# from audioldm.utils import default_audioldm_config, get_duration,\
#   get_bit_depth, get_metadata, download_checkpoint, save_wave
# from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
# from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm2 import (
    text_to_audio,
    # style_transfer,
    build_model,
    save_wave,
    # get_time,
    # round_up_duration,
    # get_duration,
    # super_resolution_and_inpainting,
)


class AudioLdmWrapper:
    def __init__(
        self,
        model_name,
        audio_model_precision,
        device="cpu",
        save_dir="./",
        random_seed=0,
    ):
        self.random_seed = random_seed
        self.is_ready_ = False
        self.device = device
        self.model = init_text2audio_model(
            model_name, torch_dtype=audio_model_precision, device=device
        )
        self.output_dir = save_dir
        self.is_ready_ = True
        print("AudioLDM model init done")

    def text_to_audio(
        self,
        text,
        duration=10,  # in seconds
        infer_steps=100,
        num_candidate=1,
        # guidance_scale,
        save_name=None,
    ):
        if not self.is_ready_:
            return False
        self.is_ready_ = False
        waveforms = self.model(
            text,
            audio_length_in_s=duration,
            num_inference_steps=infer_steps,
            num_waveforms_per_prompt=num_candidate,  # mps device not support >= 1
        ).audios

        if save_name is not None:
            for i, audio in enumerate(waveforms, 0):
                save_path = f"{self.output_dir}{save_name}_{i}.wav"
                wavfile.write(save_path, rate=srate, data=audio)
        self.is_ready_ = True
        return waveforms


"""
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
"""


# @st.cache_resource
def init_img2text_model(img2text_model_name, model_backend, device="cpu"):
    if model_backend == "huggingface":
        from transformers import pipeline

        pipe = pipeline("image-to-text", model=img2text_model_name)

    elif model_backend == "modelscope":
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        pipe = pipeline(Tasks.image_captioning, model=img2text_model_name)

    return pipe.to(device)


def init_text2audio_model(
    text2audio_model_name, torch_dtype=torch.float16, device="cpu"
):
    from diffusers import AudioLDM2Pipeline
    from diffusers import DPMSolverMultistepScheduler

    pipe = AudioLDM2Pipeline.from_pretrained(
        text2audio_model_name, torch_dtype=torch_dtype
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe.to(device)


@st.cache_resource
def init_model(
    img_model_name="nlpconnect/vit-gpt2-image-captioning",
    img_model_end="huggingface",
    audio_model_name="cvssp/audioldm2",
    audio_model_precision=torch.float16,
    device="cpu",
    save_dir="./",
):
    image_tool = init_img2text_model(img_model_name, img_model_end, device)
    # audio_tool = AudioLdmWrapper(audio_model_name, audio_model_ckpt)
    audio_tool = AudioLdmWrapper(
        audio_model_name,
        audio_model_precision=audio_model_precision,
        save_dir=save_dir,
        device=device,
    )
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
    from audioldm2.pipeline import text_to_audio

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
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    save_dir = "./output/"
    from diffusers import AudioLDM2Pipeline

    model_id = "cvssp/audioldm2"
    audio_tool = AudioLdmWrapper(
        model_id, audio_model_precision=torch.float16, save_dir=save_dir, device=device
    )
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    prompt = "Castle in the sky"
    srate = 16000
    save_name = "test"
    duration = 5
    infer_steps = 20
    num_candidate = 1
    audio = audio_tool.text_to_audio(
        prompt,
        duration=duration,
        infer_steps=infer_steps,
        num_candidate=num_candidate,
        save_name=save_name,
    )