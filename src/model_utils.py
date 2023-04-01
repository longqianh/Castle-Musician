import streamlit as st
import torch
import yaml
from audioldm import LatentDiffusion, seed_everything
from audioldm.utils import default_audioldm_config, get_duration,\
      get_bit_depth, get_metadata, download_checkpoint, save_wave
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from audioldm.latent_diffusion.ddim import DDIMSampler

@st.cache_resource
def init_img2text_model(img2text_model_path,model_backend):
    
    if model_backend == 'huggingface':
        from transformers import pipeline
        img2text_model = pipeline("image-to-text", model=img2text_model_path)
        # tokenizer=img2text_model_path
    elif model_backend == 'modelscope':
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        from modelscope.outputs import OutputKeys

        img2text_model = pipeline(Tasks.image_captioning, model=img2text_model_path)

    return img2text_model

@st.cache_resource
def init_text2audio_model(ckpt_path=None,config=None):
        
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()
    
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    text2audio_model = LatentDiffusion(**config["model"]["params"])

    checkpoint = torch.load(ckpt_path, map_location=device)
    text2audio_model.load_state_dict(checkpoint["state_dict"])

    text2audio_model.eval()
    text2audio_model = text2audio_model.to(device)

    text2audio_model.cond_stage_model.embed_mode = "text"
    return text2audio_model

@st.cache_resource
def init_audio_refine_model(audio_refine_model_ckpt):
    # audio superresolution
    return audio_refine_model

def init_model():
    img2text_model = init_img2text_model()
    text2audio_model = init_text2audio_model()
    audio_refine_model = init_audio_refine_model()
    return img2text_model, text2audio_model, audio_refine_model


def predict_audio_from_image(img, model):
    pass

def predict_audio_from_text(text, model, seed=42,duration=10,ddim_steps=200,\
                            guidance_scale=2.5,n_candidate_gen_per_text=3):  
    from audioldm.pipeline import text_to_audio
    return text_to_audio(model, text, duration=duration, seed=seed,ddim_steps=ddim_steps,\
                         guidance_scale=guidance_scale,n_candidate_gen_per_text=n_candidate_gen_per_text)

def predict_text_from_image(img, model):
    img_caption=model(img)

    return img_caption.detach().cpu().numpy()


if __name__ == "__main__":
    save_path='./output/'
    audioldm=init_text2audio_model('./data/ldm_trimmed.ckpt')
    audio=predict_audio_from_text('test',audioldm,duration=1,ddim_steps=20,\
                             guidance_scale=1,n_candidate_gen_per_text=1)
    srate=16000
    save_wave(audio, save_path, samplerate=srate, name='test')