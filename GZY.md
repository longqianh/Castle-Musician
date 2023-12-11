


```
conda create -n hlq python=3.8
conda activate hlq

### pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

pip install jupyter tqdm soundfile progressbar einops scipy librosa librosa==0.9.2 torchlibrosa transformers ftfy pandas -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

pip install opencv-python timm unicodedata2 zhconv decord>=0.6.0 rapidfuzz -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

pip install scipy wave noisereduce -i https://pypi.tuna.tsinghua.edu.cn/simple some-package


pip install streamlit audioldm2 modelscope
pip install modelscope

```