# Chatterbox TTS — Persian / فارسی

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-Weights_on_HF-yellow)](https://huggingface.co/amirzandi/Chatterbox-Persian-TTS)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12OU6ynF9b7gOVSRn2HHG8dLgSwBPRIkt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A **full fine-tune (FFT)** of the [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) T3 backbone on Persian speech:
**109,400 clips / 417 hours** of [Thomcles/Persian-Farsi-Speech](https://huggingface.co/datasets/Thomcles/Persian-Farsi-Speech).

Raw Persian text in, speech out. **No G2P, no phonemes** — the official multilingual tokenizer already covers Persian graphemes.
> Persian text-to-speech model — provide raw text, get speech. Details and limitations: [Model card on Hugging Face](https://huggingface.co/amirzandi/Chatterbox-Persian-TTS).
**The weights (~2.14GB) live on Hugging Face** → [amirzandi/Chatterbox-Persian-TTS](https://huggingface.co/amirzandi/Chatterbox-Persian-TTS) — this repo hosts the notebook and references (GitHub LFS free quota cannot host the checkpoint).

## 🎧 Ready-made voices — then make your own

Two ready samples (also embedded in the Colab notebook):

| Sample | Text |
|---|---|
<audio controls src="https://huggingface.co/amirzandi/Chatterbox-Persian-TTS/resolve/main/demo/voice1.wav">Your browser does not support audio.</audio> | «صبح زود، وقتی از خانه بیرون آمدم، متوجه شدم هوای شهر برخلاف روزهای گذشته آرام و دل‌پذیر است. در مسیر، یک صدای غیرعادی از خیابان کناری شنیدم و برای چند لحظه مکث کردم، اما ناگهانی اتفاقی افتاد که باعث شد همه‌ی رهگذران به سمت دیگری نگاه کنند. من بدون عجله به راهم ادامه دادم و سعی کردم لحن صحبت کردنم را طبیعی و آرام نگه دارم، چون حتی یک تغییر کوچک در تلفظ یا مکث می‌تواند کیفیت گفتار را تحت تأثیر قرار دهد.» |
| <audio controls src="https://huggingface.co/amirzandi/Chatterbox-Persian-TTS/resolve/main/demo/voice2.wav">Your browser does not support audio.</audio> | «این یک پاراگراف بلندتره تا سیستم چانکینگ رو تست کنه و کلمه‌ها رو درست سر هم بذاره.» |

Anything else? **Generate it yourself** in the ready Colab notebook (zero config, best-of-N retry, Whisper verification) — free GPU:

**→ [Open the inference notebook on Colab](https://colab.research.google.com/drive/12OU6ynF9b7gOVSRn2HHG8dLgSwBPRIkt)**

The same notebook file lives in this repo: [farsi_tts_fft_lora_inference.ipynb](farsi_tts_fft_lora_inference.ipynb).

## 💻 Minimal usage

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from safetensors.torch import load_file

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
sd = load_file("t3_persian_v1.safetensors")
model.t3.load_state_dict(
    {k[3:] if k.startswith("t3.") else k: v for k, v in sd.items()}, strict=False)
model.t3.to("cuda").eval()

wav = model.generate(
    punc_norm("سلام، امروز هوا خوب است."),
    language_id=None,
    audio_prompt_path="ref.wav",   # any clean 5-10s Persian voice clip
    temperature=0.6, repetition_penalty=1.5,
)
```

Recommended: `temperature=0.6`, `repetition_penalty=1.5`, and 2–3 takes per sentence (occasional hallucination is inherent to the base architecture; a retry usually fixes it).

## 📊 Results (Whisper large-v3-turbo CER, best-of-8)

- Natural Persian text (incl. long paragraphs): **CER 0.01–0.09**
- 22-sentence stress set (big numbers, clock times, foreign words, tongue twisters): mean CER **0.038–0.062**
- Known weak spot: artificial tongue-twisters can slide (CER ≈ 0.3 after 8 tries)

The released checkpoint is the most stable point of the training run; later checkpoints overfit.

## ⚠️ Honest limitations

16kHz output · output quality depends on the reference voice · occasional hallucination (retry) · numbers mostly work but are not guaranteed · foreign words are read but not natively pronounced · trained on 417h of general speech — it improves Persian voice cloning, it does not reproduce one specific speaker. Full list on the [model card](https://huggingface.co/amirzandi/Chatterbox-Persian-TTS).

## 📁 Repository contents

| File | Status |
|---|---|
| `farsi_tts_fft_lora_inference.ipynb` | **current** — daily inference (FFT or LoRA), chunking, best-of-N retry, Whisper check. Also on [Colab](https://colab.research.google.com/drive/12OU6ynF9b7gOVSRn2HHG8dLgSwBPRIkt). |
| `farsi_tts.py` | ⚠️ **legacy** — helper from an earlier experiment. It loads a `final_model/` folder and is **not compatible** with the released weights. Kept for reference. |
| `g2p_exceptions.json` | legacy, used only by `farsi_tts.py`. |

The full training pipeline (preprocess → cache → FFT/LoRA → eval) runs on Kaggle ([training output dataset](https://www.kaggle.com/datasets/amirhosseinzandi/farsi-tts-train-thom-mtl-output)).

## 📄 License & credits

- Fine-tuned weights: **MIT** — any use permitted, with attribution
- Base model: [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (MIT)
- Training data: [Thomcles/Persian-Farsi-Speech](https://huggingface.co/datasets/Thomcles/Persian-Farsi-Speech) (Apache-2.0) — special thanks to **Thomcles** for releasing 417h of Persian speech
