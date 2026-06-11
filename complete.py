import os
import sys
import subprocess
import logging
import threading
import uuid
import time
import shutil
import re

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- ML Imports ---
import torch
import whisper
from gtts import gTTS
from transformers import (
    pipeline, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    MarianTokenizer, 
    MarianMTModel
)
from IndicTransToolkit.processor import IndicProcessor

# ==========================================
# 1. SERVER INITIALIZATION & CONFIG
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DubSync")

app = Flask(__name__)
CORS(app) 
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100MB Max Upload

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# State Management
jobs = {}

# ==========================================
# 2. GLOBAL ML MODEL INITIALIZATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Initializing Unified ML Engine on device: {DEVICE.upper()}")

# --- A. Load Whisper ---
logger.info("Loading Whisper Model (Small)...")
whisper_model = whisper.load_model("small", device=DEVICE)

# --- B. Load Korean Model ---
logger.info("Loading Korean Model (ke-t5-base)...")
korean_translator = pipeline(
    "translation",
    model="seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko",
    device=0 if DEVICE == "cuda" else -1
)

# --- C. Load IndicTrans2 Model ---
logger.info("Loading IndicTrans2 Model (200M)...")
INDIC_MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
indic_tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL_NAME, trust_remote_code=True)
indic_model = AutoModelForSeq2SeqLM.from_pretrained(
    INDIC_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
indic_model.eval()
ip = IndicProcessor(inference=True)

# Warmup Indic Model
logger.info("Warming up Indic model...")
try:
    warmup_batch = ip.preprocess_batch(["Hello"], src_lang="eng_Latn", tgt_lang="hin_Deva")
    warmup_inputs = indic_tokenizer(warmup_batch, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        _ = indic_model.generate(**warmup_inputs, max_length=50, num_beams=1, use_cache=False)
    logger.info("Models loaded and warmed up successfully!")
except Exception as e:
    logger.warning(f"Warmup failed: {e}")

# --- D. Dynamic MarianMT Storage ---
loaded_marian_models = {}

# ==========================================
# 3. LANGUAGE ROUTING DICTIONARY
# ==========================================
languages = {
    # MarianMT Languages
    "es": {"name": "Spanish", "engine": "marian", "model": "Helsinki-NLP/opus-mt-en-es", "gtts_lang": "es", "font": "Arial"},
    "fr": {"name": "French", "engine": "marian", "model": "Helsinki-NLP/opus-mt-en-fr", "gtts_lang": "fr", "font": "Arial"},
    
    # Korean Pipeline
    "ko": {"name": "Korean", "engine": "korean", "gtts_lang": "ko", "font": "NanumGothic"},
    
    # IndicTrans2 Languages
    "hi": {"name": "Hindi", "engine": "indic", "indic_tag": "hin_Deva", "gtts_lang": "hi", "font": "Noto Sans Devanagari"},
    "ta": {"name": "Tamil", "engine": "indic", "indic_tag": "tam_Taml", "gtts_lang": "ta", "font": "Noto Sans Tamil"},
    "te": {"name": "Telugu", "engine": "indic", "indic_tag": "tel_Telu", "gtts_lang": "te", "font": "Noto Sans Telugu"},
    "kn": {"name": "Kannada", "engine": "indic", "indic_tag": "kan_Knda", "gtts_lang": "kn", "font": "Noto Sans Kannada"},
    "ml": {"name": "Malayalam", "engine": "indic", "indic_tag": "mal_Mlym", "gtts_lang": "ml", "font": "Noto Sans Malayalam"},
    "gu": {"name": "Gujarati", "engine": "indic", "indic_tag": "guj_Gujr", "gtts_lang": "gu", "font": "Noto Sans Gujarati"},
    "bn": {"name": "Bengali", "engine": "indic", "indic_tag": "ben_Beng", "gtts_lang": "bn", "font": "Noto Sans Bengali"},
    "mr": {"name": "Marathi", "engine": "indic", "indic_tag": "mar_Deva", "gtts_lang": "mr", "font": "Noto Sans Devanagari"},
    "pa": {"name": "Punjabi", "engine": "indic", "indic_tag": "pan_Guru", "gtts_lang": "pa", "font": "Noto Sans Gurmukhi"},
    "ur": {"name": "Urdu", "engine": "indic", "indic_tag": "urd_Arab", "gtts_lang": "ur", "font": "Noto Nastaliq Urdu"}
}

# ==========================================
# 4. UNIFIED TRANSLATION ROUTER
# ==========================================
def unified_translate(valid_chunks, lang_code):
    """Routes chunks to the correct ML model loaded in memory"""
    if not valid_chunks: return []
    
    lang_config = languages[lang_code]
    engine = lang_config["engine"]
    
    try:
        # --- ROUTE 1: INDIC LANGUAGES ---
        if engine == "indic":
            tgt_tag = lang_config["indic_tag"]
            logger.info(f"Routing {len(valid_chunks)} chunks to IndicTrans2 ({tgt_tag})")
            
            # Sub-batching for safety
            all_translations = []
            for i in range(0, len(valid_chunks), 4):
                batch = valid_chunks[i:i+4]
                prep_batch = ip.preprocess_batch(batch, src_lang="eng_Latn", tgt_lang=tgt_tag)
                inputs = indic_tokenizer(prep_batch, truncation=True, padding="longest", max_length=256, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    gen_tokens = indic_model.generate(**inputs, use_cache=False, max_length=128, num_beams=2, early_stopping=True)
                
                decoded = indic_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                post_batch = ip.postprocess_batch(decoded, lang=tgt_tag)
                all_translations.extend(post_batch)
                
            return all_translations

        # --- ROUTE 2: KOREAN ---
        elif engine == "korean":
            logger.info(f"Routing {len(valid_chunks)} chunks to Korean T5")
            results = korean_translator(valid_chunks)
            return [r["translation_text"] for r in results]

        # --- ROUTE 3: MARIAN MT (French/Spanish) ---
        elif engine == "marian":
            model_url = lang_config["model"]
            logger.info(f"Routing {len(valid_chunks)} chunks to MarianMT ({model_url})")
            
            if model_url not in loaded_marian_models:
                logger.info(f"Loading {model_url} into memory...")
                m_tok = MarianTokenizer.from_pretrained(model_url)
                m_mod = MarianMTModel.from_pretrained(model_url).to(DEVICE).eval()
                loaded_marian_models[model_url] = {"tokenizer": m_tok, "model": m_mod}
                
            m_data = loaded_marian_models[model_url]
            m_tok, m_mod = m_data["tokenizer"], m_data["model"]
            
            all_translations = []
            for i in range(0, len(valid_chunks), 8):
                batch = valid_chunks[i:i+8]
                inputs = m_tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.inference_mode():
                    outputs = m_mod.generate(**inputs, max_new_tokens=256)
                decoded = [m_tok.decode(out, skip_special_tokens=True) for out in outputs]
                all_translations.extend(decoded)
                
            return all_translations

    except Exception as e:
        logger.error(f"Translation routing failed: {e}")
        raise e

# ==========================================
# 5. MEDIA HELPERS
# ==========================================
def get_duration(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(result.stdout.strip())
        return dur if dur > 0 else 0.01
    except:
        return 0.01

def format_srt_timestamp(t):
    t = max(0.0, float(t))
    h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(segments, texts, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg, txt in zip(segments, texts):
            txt = (txt or "").strip()
            if not txt: continue
            start = format_srt_timestamp(seg["start"])
            end = format_srt_timestamp(seg["end"])
            line = " ".join(txt.split())
            f.write(f"{idx}\n{start} --> {end}\n{line}\n\n")
            idx += 1

def convert_srt_to_ass_with_fonts(srt_file, ass_file, font_name):
    try:
        with open(srt_file, 'r', encoding='utf-8') as f: srt_content = f.read()
        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginV
Style: Default,{font_name},48,&H00FFFFFF,&H00000000,&H80000000,0,1,3,1,2,30

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        def srt_time_to_ass(st):
            st = st.replace(',', '.')
            h, m, s = st.split(':')
            return f"{int(h)}:{int(m):02d}:{float(s):05.2f}"
            
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_header)
            blocks = re.split(r'\n\n+', srt_content.strip())
            for block in blocks:
                lines = block.split('\n')
                if len(lines) >= 3:
                    match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                    if match:
                        text = ' '.join(lines[2:]).replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
                        f.write(f"Dialogue: 0,{srt_time_to_ass(match.group(1))},{srt_time_to_ass(match.group(2))},Default,,0,0,0,,{text}\n")
        return ass_file
    except:
        return None

# ==========================================
# 6. THE CORE PIPELINE
# ==========================================
def run_translation_pipeline(job_id, input_video_path, file_basename, lang_code):
    try:
        logger.info(f"[Job {job_id}] Starting processing pipeline...")
        selected_lang = languages[lang_code]
        gtts_code = selected_lang["gtts_lang"]
        font_name = selected_lang.get("font", "Arial")

        # Define all paths
        t_audio = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{file_basename}.mp3")
        o_audio = os.path.join(app.config['UPLOAD_FOLDER'], f"out_{gtts_code}_{file_basename}.mp3")
        s_audio = os.path.join(app.config['UPLOAD_FOLDER'], f"stretch_{gtts_code}_{file_basename}.mp3")
        srt_file = os.path.join(app.config['UPLOAD_FOLDER'], f"subs_{gtts_code}_{file_basename}.srt")
        ass_file = os.path.join(app.config['UPLOAD_FOLDER'], f"subs_{gtts_code}_{file_basename}.ass")
        
        final_video_name = f"final_{gtts_code}_{file_basename}.mp4"
        final_video_path = os.path.join(app.config['PROCESSED_FOLDER'], final_video_name)
        
        final_srt_name = f"final_{gtts_code}_{file_basename}.srt"
        final_srt_path = os.path.join(app.config['PROCESSED_FOLDER'], final_srt_name)

        # 1. EXTRACT
        jobs[job_id].update({"step": "extracting_audio", "progress": 10})
        subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "192k", t_audio], check=True, capture_output=True)

        # 2. TRANSCRIBE 
        jobs[job_id].update({"step": "transcribing", "progress": 25})
        result = whisper_model.transcribe(t_audio, language="en")
        segments = result.get("segments", [])
        eng_texts = [s.get("text", "").strip() for s in segments if s.get("text")]
        if not eng_texts: raise Exception("No valid speech detected.")

        # 3. TRANSLATE (Internal Call)
        jobs[job_id].update({"step": "translating", "progress": 45})
        translated_texts = unified_translate(eng_texts, lang_code)
        
        min_len = min(len(eng_texts), len(translated_texts))
        translated_texts = translated_texts[:min_len]
        segments = segments[:min_len]

        # 4. GENERATE TTS
        jobs[job_id].update({"step": "generating_audio", "progress": 65})
        tts = gTTS(text=" ".join(translated_texts), lang=gtts_code, slow=False)
        tts.save(o_audio)

        # 5. STRETCH
        jobs[job_id].update({"step": "syncing_audio", "progress": 75})
        v_dur = get_duration(input_video_path)
        a_dur = get_duration(o_audio)
        ratio = a_dur / (v_dur if v_dur > 0 else 0.01)
        
        filter_str = f"atempo={ratio:.3f}" if 0.5 <= ratio <= 2.0 else (f"atempo=0.5,atempo={ratio/0.5:.3f}" if ratio < 0.5 else f"atempo=2.0,atempo={ratio/2.0:.3f}")
        subprocess.run(["ffmpeg", "-y", "-i", o_audio, "-filter:a", filter_str, s_audio], check=True, capture_output=True)

        # 6. GENERATE SUBTITLES
        jobs[job_id].update({"step": "creating_subtitles", "progress": 85})
        write_srt(segments, translated_texts, srt_file)
        if os.path.exists(srt_file): shutil.copy(srt_file, final_srt_path)

        # 7. MERGE VIDEO
        jobs[job_id].update({"step": "merging_video", "progress": 90})
        has_subs = os.path.exists(srt_file) and os.path.getsize(srt_file) > 10
        video_created = False

        if has_subs:
            ass_path = convert_srt_to_ass_with_fonts(srt_file, ass_file, font_name)
            if ass_path:
                try:
                    p = ass_path.replace('\\', '/')
                    subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-i", s_audio, "-vf", f"ass='{p}'", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-c:a", "aac", "-b:a", "192k", final_video_path], check=True, capture_output=True, timeout=600)
                    video_created = True
                except: logger.warning("ASS burn failed. Trying SRT...")

            if not video_created:
                try:
                    p = os.path.abspath(srt_file).replace('\\', '/')
                    style = f"FontName={font_name},FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=30"
                    subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-i", s_audio, "-vf", f"subtitles='{p}':force_style='{style}'", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-c:a", "aac", "-b:a", "192k", final_video_path], check=True, capture_output=True, timeout=600)
                    video_created = True
                except: logger.warning("SRT burn failed. Trying Soft Subs...")

            if not video_created:
                try:
                    subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-i", s_audio, "-i", srt_file, "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0", "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text", "-metadata:s:s:0", f"language={gtts_code}", final_video_path], check=True, capture_output=True)
                    video_created = True
                except: logger.warning("Soft subs failed. Merging audio only...")

        if not video_created:
            subprocess.run(["ffmpeg", "-y", "-i", input_video_path, "-i", s_audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", final_video_path], check=True, capture_output=True)

        if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) < 1000:
            raise Exception("Final video generation failed.")

        # FINAL VERDICT
        jobs[job_id].update({
            "status": "complete",
            "progress": 100,
            "videoUrl": f"/{app.config['PROCESSED_FOLDER']}/{final_video_name}",
            "subtitleUrl": f"/{app.config['PROCESSED_FOLDER']}/{final_srt_name}" if has_subs else None
        })
        logger.info(f"[Job {job_id}] SUCCESS")

    except Exception as e:
        logger.error(f"[Job {job_id}] ERROR: {e}")
        jobs[job_id].update({"status": "error", "error": str(e)})

    finally:
        for f in [t_audio, o_audio, s_audio, srt_file, ass_file, input_video_path]:
            if os.path.exists(f): 
                try: os.remove(f)
                except: pass

# ==========================================
# 7. FLASK ROUTES
# ==========================================
@app.route("/api/process-video", methods=["POST"])
def process_video_request():
    if 'video' not in request.files: return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    lang_code = request.form.get('language')
    
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if not lang_code or lang_code not in languages: return jsonify({"error": "Invalid language"}), 400

    safe_name = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    file.save(in_path)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0, "step": "starting", "videoUrl": None, "subtitleUrl": None, "error": None}

    threading.Thread(target=run_translation_pipeline, args=(job_id, in_path, os.path.splitext(safe_name)[0], lang_code)).start()
    return jsonify({"taskId": job_id}), 202

@app.route("/api/check-status")
def check_status():
    job_id = request.args.get('id')
    if not job_id or job_id not in jobs: return jsonify({"error": "Invalid job ID"}), 404
    return jsonify(jobs[job_id])

@app.route('/static/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route("/api/languages")
def get_languages():
    return jsonify([{"code": k, "name": v["name"]} for k, v in languages.items()])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DubSync v2.0 - Unified EC2 Monolith")
    print("="*60)
    print(f" Hardware Detected: {DEVICE.upper()}")
    print(" All ML Models Loaded into Local Memory")
    print(" External API Dependencies (Ngrok) Removed")
    print(f" Local Endpoint: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, use_reloader=False, port=5000)