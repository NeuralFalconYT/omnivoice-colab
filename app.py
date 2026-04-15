# %cd /content/omnivoice-colab
import os
import sys
import logging
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import os
import uuid
temp_audio_dir="./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Setup path to import subtitle_maker from /content/omnivoice-colab/OmniVoice/
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

# Attempt to import Whisper's supported language dict to filter unsupported languages
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
# ---------------------------------------------------------------------------
print("Loading model from k2-fsa/OmniVoice to cuda ...")

from hf_mirror import download_model

try:
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
except Exception as e:
  omnivoice_model_path=download_model(
    "k2-fsa/OmniVoice",
    download_folder="./OmniVoice_Model",
    redownload=False,
    workers=6,
    use_snapshot=False,
  )

  model = OmniVoice.from_pretrained(
      omnivoice_model_path,
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

# JS for Voice Clone Tab Textbox
INSERT_TAG_JS_VC = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vc_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# JS for Voice Design Tab Textbox
INSERT_TAG_JS_VD = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vd_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# UI Configurations & Language Mappings
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent", "Australian Accent", "British Accent", "Chinese Accent",
        "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent"
    ],
    "Chinese Dialect": [
        "Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect",
        "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect",
        "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"
    ],
}

DIALECT_MAP = {
    "Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话",
    "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话",
    "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话",
    "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话",
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
    "Chinese Dialect": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    """Check if the selected language is supported by Whisper to save processing time."""
    if not lang or lang == "Auto":
        return True 
    
    if WHISPER_LANGUAGE_CODE is None:
        return True 
    
    supported_langs = [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] + \
                      [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    
    lang_lower = lang.lower()
    for w_lang in supported_langs:
        if w_lang in lang_lower or lang_lower in w_lang:
            return True
            
    return False

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    """Generates Subtitles only if user requested them and language is supported."""
    if not want_subs:
        return None, None, None

    if not _is_whisper_supported(lang):
        logging.warning(f"Language '{lang}' is likely unsupported by Whisper. Skipping subtitle generation.")
        return None, None, None

    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3] 
    except Exception as e:
        logging.warning(f"Subtitle generation failed: {e}")

    return None, None, None


def tts_file_name(text, language="en"):
    global temp_audio_dir

    # --- Clean text ---
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters + spaces
    clean_text = clean_text.lower().strip().replace(" ", "_")

    if not clean_text:
        clean_text = "audio"

    # --- Truncate ---
    truncated = clean_text[:20]

    # --- Clean language ---
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"

    # --- Random suffix ---
    rand = uuid.uuid4().hex[:8].upper()

    # --- Final filename ---
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"


def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale, 
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None
):
    """Core Text-to-Speech Generation Logic"""
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    if mode == "clone" and ref_audio and not ref_text:
        try:
            whisper_lang = language if (language and language != "Auto") else None
            whisper_results = subtitle_maker(ref_audio, whisper_lang)
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logging.warning(f"Fallback transcription failed: {e}")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    # waveform = audio[0].squeeze(0).numpy()
    # waveform = (waveform * 32767).astype(np.int16)
    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

/* CSS for Event Tags */
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=None, label="Duration (seconds)", info="Set a fixed duration to override speed.")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="Lower = faster, higher = better quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True, info="Applies silence removal and trims reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True, info="Removes long silences from generated audio.")
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice Multilingual </h1>
            <p>State-of-the-art text-to-speech model for 600+ languages, supporting Voice Clone and Voice Design.</p>
        </div>
    """)

    with gr.Tabs():
        # ==============================================================
        # Voice Clone Tab
        # ==============================================================
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Added elem_id for JS hook
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vc_textbox")
                    
                    # Tag Buttons for Voice Clone
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vc_text],
                                outputs=vc_text,
                                js=INSERT_TAG_JS_VC
                            )

                    with gr.Row():
                      vc_lang = _lang_dropdown("Language (optional)")
                      vc_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio (3–10 seconds audio)", type="filepath", elem_classes="compact-audio")
                    
                    vc_ref_text = gr.Textbox(
                        label="Reference Text", lines=2, 
                        placeholder="Auto-transcribed upon audio upload. You can manually edit it if Whisper gets it wrong."
                    )
                                        
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="Generated Audio (WAV)")
                        vc_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vc_out_word_srt = gr.File(label="Word Level SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            def _auto_transcribe(audio_path, lang):
                if not audio_path:
                    return gr.update(value="")
                try:
                    whisper_lang = lang if lang != "Auto" else None
                    whisper_results = subtitle_maker(audio_path, whisper_lang)
                    if whisper_results and len(whisper_results) > 7:
                        return gr.update(value=whisper_results[7])
                except Exception as e:
                    logging.warning(f"Auto-transcription failed: {e}")
                return gr.update(value="")

            vc_ref_audio.change(
                fn=_auto_transcribe,
                inputs=[vc_ref_audio, vc_lang],
                outputs=[vc_ref_text]
            )

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None:
                    return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                # tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                tmp_wav=tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        # ==============================================================
        # Voice Design Tab
        # ==============================================================
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Added elem_id for JS hook
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vd_textbox")
                    
                    # Tag Buttons for Voice Design
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vd_text],
                                outputs=vd_text,
                                js=INSERT_TAG_JS_VD
                            )

                    with gr.Row():
                      vd_lang = _lang_dropdown(value='Auto')
                      vd_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Auto"
                            if _cat == "Gender":
                                default_val = "Female"
                            elif _cat == "Age":
                                default_val = "Young Adult"
                                
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=["Auto"] + _choices, value=default_val, info=_ATTR_INFO.get(_cat))
                            )
                        
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="Generated Audio (WAV)")
                        vd_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vd_out_word_srt = gr.File(label="Word Level SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                if not selected: return None
                return ", ".join([DIALECT_MAP.get(v, v) for v in selected])

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                if res[0] is None:
                    return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav=tts_file_name(text, language=lang)
                # tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                wavfile.write(tmp_wav, sr, waveform)
                
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
