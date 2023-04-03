# Castle Musician
Castle musician is an AIGC musician project designed for high-quality music generation. The ultimate goal is to use AIGC-created music on patients with fMRI or fNIRS signal monitoring.



### Startup

`streamlit run app.py`



### Requirements

- Pydub for audio handling: `pip install pydub` and `sudo install ffmpeg`

- AudioLDM for audio generation: `pip install audioldm`

- Huggingface Transformers for image captioning: `pip install transformers`

- Modelscope for image captioning: `pip install opencv-python, timm, zhconv, unicodedata2, rapidfuzz, modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple `

- StreamLit for web app: `pip install streamlit`

- AudioLDM env: 
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

pip install jupyter tqdm soundfile progressbar einops scipy librosa librosa==0.9.2 torchlibrosa transformers ftfy pandas -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

- new image model
```
pip install opencv-python timm unicodedata2 zhconv decord>=0.6.0 rapidfuzz

```

- audio noise reduce function
```
pip install scipy wave noisereduce -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

```

---

### DEBUG record
- sample_rate=16000 ； 
- transfer_strength不要超过1.0； 
- 网页的音频结果换回读文件的方式(之前的numpy格式不是它需要的)； 
- 新的图像模型输出字典key和之前的模型不同； 
  

  



