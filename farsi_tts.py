"""
رابط ساده و فانکشنال (بدون کلاس) برای مدل فاین‌تیون‌شده Persian Chatterbox TTS —
شامل تمام قابلیت‌های نوت‌بوک تست: chunking متن‌های بلند، retry روی قطع‌شدن زودهنگام
یا نتیجه‌ی غلط، verification با Whisper، و merge کردن g2p_exceptions موقت.

پیش‌نیاز: این فایل باید داخل پوشه chatterbox-finetuning اجرا بشه (importهای
src.chatterbox_ و g2p_utils نسبی هستن).

استفاده‌ی ساده (مثل Piper):
    import os
    os.chdir("chatterbox-finetuning")
    from farsi_tts import load_model, synthesize

    model = load_model("final_model")
    synthesize(model, "سلام به همگی!", "test.wav")

استفاده‌ی کامل با همه‌ی گزینه‌ها:
    model = load_model(
        "final_model",
        g2p_exceptions={"مکث": "m a k s e"},
        verification=True,
    )
    result = synthesize(
        model,
        "یک متن طولانی با چند جمله ...",
        "out.wav",
        temperature=0.5,
        repetition_penalty=1.5,
    )
    print(result["ok"], result["num_chunks"])
"""

import os
import re
import json
import shutil
import logging

import torch
import torchaudio as ta

_CER_PUNCT_RE = re.compile(r"[.,،؛:؟!…\-]")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!؟…])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[،؛:])\s+")

_captured_logs = []


class _CaptureHandler(logging.Handler):
    def emit(self, record):
        _captured_logs.append(record.getMessage())


# ---------------------------------------------------------------- load_model

def load_model(
    final_model_dir,
    audio_prompt_path=None,
    device=None,
    g2p_exceptions=None,
    verification=True,
    whisper_model="openai/whisper-small",
):
    """
    مدل رو یک‌بار لود می‌کنه و یک دیکشنری برمی‌گردونه که به synthesize() پاس می‌دی.

    final_model_dir : مسیر پوشه final_model
    audio_prompt_path : اگه بخوای به‌جای صدای داخل final_model از یک wav دیگه
        استفاده کنی؛ پیش‌فرض None یعنی از voice_reference.wav داخل خودش استفاده کن
    g2p_exceptions : دیکشنری اختیاری {کلمه فارسی: تلفظ آوایی} که روی
        g2p_exceptions.json بسته‌شده داخل final_model اضافه (merge) می‌شه
    verification : اگه True باشه هر تکه‌ی صدا با Whisper چک می‌شه که درست تولید
        شده یا نه؛ برای سرعت بیشتر می‌تونی False بذاری
    """
    from src.chatterbox_.tts import ChatterboxTTS
    from src.chatterbox_.models.t3.t3 import T3
    from src.config import TrainConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(final_model_dir, "export_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    # g2p_utils.py و g2p_exceptions.json باید کنار همین اسکریپت (working dir) باشن
    src_g2p = os.path.join(final_model_dir, "g2p_utils.py")
    if not os.path.exists("g2p_utils.py") and os.path.exists(src_g2p):
        shutil.copy(src_g2p, "g2p_utils.py")

    exc_src = os.path.join(final_model_dir, "g2p_exceptions.json")
    exc_dst = "g2p_exceptions.json"
    if os.path.exists(exc_src) and not os.path.exists(exc_dst):
        shutil.copy(exc_src, exc_dst)

    if g2p_exceptions:
        exceptions = {"text_substitutions": {}, "phonemic_overrides": {}}
        if os.path.exists(exc_dst):
            with open(exc_dst, encoding="utf-8") as f:
                exceptions = json.load(f)
        exceptions.setdefault("phonemic_overrides", {}).update(g2p_exceptions)
        with open(exc_dst, "w", encoding="utf-8") as f:
            json.dump(exceptions, f, ensure_ascii=False, indent=2)

    from g2p_utils import to_phonemic, normalize_only

    cfg = TrainConfig()
    model = ChatterboxTTS.from_local(cfg.model_dir, device=device)

    hp = model.t3.hp
    hp.text_tokens_dict_size = manifest["text_tokens_dict_size"]
    hp.use_cache = False

    final_t3 = T3(hp=hp).to(device)
    final_t3.load_state_dict(
        torch.load(os.path.join(final_model_dir, "t3_merged.pt"), map_location=device)
    )
    model.t3 = final_t3
    model.t3.eval()

    if audio_prompt_path is None and manifest.get("has_voice_reference"):
        audio_prompt_path = os.path.join(final_model_dir, "voice_reference.wav")

    # برای تشخیص forced-EOS (قطع‌شدن زودهنگام به‌خاطر anti-repetition guard)
    logging.getLogger(
        "src.chatterbox_.models.t3.inference.alignment_stream_analyzer"
    ).addHandler(_CaptureHandler())

    asr = None
    if verification:
        from transformers import pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device=0 if device == "cuda" else -1,
            generate_kwargs={"language": "fa", "task": "transcribe"},
        )

    return {
        "model": model,
        "audio_prompt": audio_prompt_path,
        "to_phonemic": to_phonemic,
        "normalize_only": normalize_only,
        "asr": asr,
        "device": device,
    }


# ------------------------------------------------------------ helper: کیفیت

def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _compute_cer(ref, hyp):
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _clean_for_cer(state, text):
    t = state["normalize_only"](text)
    t = _CER_PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _transcribe(state, wav_tensor):
    model = state["model"]
    audio = wav_tensor
    if model.sr != 16000:
        audio = ta.functional.resample(wav_tensor, model.sr, 16000)
    audio_np = audio.squeeze().cpu().numpy()
    return state["asr"]({"array": audio_np, "sampling_rate": 16000})["text"]


# --------------------------------------------------------- helper: chunking

def _split_into_chunks(text, max_words=22, min_words=3):
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    for sent in sentences:
        if len(sent.split()) <= max_words:
            chunks.append(sent)
            continue
        pieces = [p.strip() for p in _CLAUSE_SPLIT.split(sent) if p.strip()]
        buf = ""
        for p in pieces:
            candidate = f"{buf} {p}".strip() if buf else p
            if len(candidate.split()) <= max_words:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)

    merged = []
    for c in chunks:
        if merged and len(c.split()) < min_words:
            merged[-1] = f"{merged[-1]} {c}".strip()
        else:
            merged.append(c)
    if len(merged) > 1 and len(merged[0].split()) < min_words:
        merged[1] = f"{merged[0]} {merged[1]}".strip()
        merged = merged[1:]
    return merged


def _pause_ms_for(chunk_text, sentence_pause_ms, clause_pause_ms, split_pause_ms):
    if chunk_text[-1:] in ".!؟…":
        return sentence_pause_ms
    if chunk_text[-1:] in "،؛:":
        return clause_pause_ms
    return split_pause_ms


# --------------------------------------------------- هسته: تولید هر تکه با retry

def _generate_chunk(state, chunk_text, max_retries, max_cer, generation_kwargs):
    model = state["model"]
    phonemic = state["to_phonemic"](chunk_text)
    expected_seconds = max(len(chunk_text.split()), 1) * 0.45
    ref = _clean_for_cer(state, chunk_text)

    best_wav, ok, attempt, cer = None, False, 0, None
    for attempt in range(max_retries):
        _captured_logs.clear()
        with torch.inference_mode():
            if state["audio_prompt"]:
                wav = model.generate(
                    phonemic, audio_prompt_path=state["audio_prompt"], **generation_kwargs
                )
            else:
                wav = model.generate(phonemic, **generation_kwargs)

        duration = wav.shape[-1] / model.sr
        forced_eos = any(
            "forcing EOS" in w and "token_repetition=True" in w for w in _captured_logs
        )
        eos_ok = not (forced_eos and duration < expected_seconds * 0.6)

        content_ok = True
        if state["asr"] is not None:
            try:
                hyp = _transcribe(state, wav)
                cer = _compute_cer(ref, _clean_for_cer(state, hyp))
                content_ok = cer <= max_cer
            except Exception:
                pass  # اگه verification خطا داد، همون خروجی رو قبول کن

        ok = eos_ok and content_ok
        best_wav = wav
        if ok:
            break

    return best_wav, ok, attempt + 1, cer, phonemic


# ------------------------------------------------------------------- synthesize

def synthesize(
    state,
    text,
    out_path,
    max_retries=4,
    max_words=22,
    min_words=3,
    sentence_pause_ms=300,
    clause_pause_ms=250,
    split_pause_ms=150,
    max_cer=0.35,
    **generation_kwargs,
):
    """
    یک متن (کوتاه یا بلند) رو به گفتار تبدیل می‌کنه و در out_path ذخیره می‌کنه:
      - متن به تکه‌های جمله‌ای/بندی تقسیم می‌شه (مثل نوت‌بوک اصلی)
      - هر تکه تولید و در صورت قطع‌شدن زودهنگام (forced-EOS) یا -اگه
        verification روشن باشه- نامطابقت با Whisper، دوباره تولید می‌شه
      - همه‌ی تکه‌ها با مکث مناسب بین‌شون به هم چسبونده می‌شن

    generation_kwargs مستقیم به model.generate می‌ره، مثلاً:
        synthesize(model, text, "out.wav", temperature=0.5, repetition_penalty=1.5)

    خروجی یک دیکشنری با جزییات هر تکه (متن، phonemic، تعداد تلاش، cer) برمی‌گردونه.
    """
    chunks_text = _split_into_chunks(text, max_words=max_words, min_words=min_words)

    wav_parts, chunk_infos = [], []
    for i, chunk_text in enumerate(chunks_text):
        wav, ok, attempts, cer, phonemic = _generate_chunk(
            state, chunk_text, max_retries, max_cer, generation_kwargs
        )
        chunk_infos.append(
            {"text": chunk_text, "phonemic": phonemic, "attempts": attempts, "ok": ok, "cer": cer}
        )
        wav_parts.append(wav)
        if i < len(chunks_text) - 1:
            pause = _pause_ms_for(chunk_text, sentence_pause_ms, clause_pause_ms, split_pause_ms)
            silence = torch.zeros(1, int(state["model"].sr * pause / 1000))
            wav_parts.append(silence)

    final_wav = torch.cat(wav_parts, dim=-1) if wav_parts else torch.zeros(1, 1)
    ta.save(out_path, final_wav, state["model"].sr)

    return {
        "text": text,
        "out": out_path,
        "num_chunks": len(chunks_text),
        "ok": all(c["ok"] for c in chunk_infos) if chunk_infos else False,
        "chunks": chunk_infos,
    }


def synthesize_batch(state, texts, out_dir, **kwargs):
    """چند متن رو پشت سر هم synthesize می‌کنه؛ برای هر کدوم out_dir/out_{i}.wav می‌سازه."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, text in enumerate(texts):
        out_path = os.path.join(out_dir, f"out_{i}.wav")
        results.append(synthesize(state, text, out_path, **kwargs))
    return results
