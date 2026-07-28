import torch
import torch.nn as nn
import cv2
import os
import yt_dlp
from PIL import Image
import open_clip
from torchvision import transforms, models
from pathlib import Path
import sys

# Add parent directory to sys.path to import train_models
sys.path.append(str(Path(__file__).parent.parent))
try:
    from train_models import get_model
except ImportError:
    # Fallback if train_models is not in parent
    sys.path.append(r"c:\Users\dotsm\Desktop\mixed cartoon dataset")
    from train_models import get_model

CLASSES = ["erotism", "safe", "violence"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CLIP Model Definition ---
class CLIPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_clip_model(checkpoint_path):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(DEVICE).eval()
    
    classifier = CLIPClassifier(clip_model.visual.output_dim, len(CLASSES)).to(DEVICE)
    classifier.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    classifier.eval()
    return clip_model, classifier, clip_preprocess

def load_nasnet_model(checkpoint_path):
    model = get_model("nasnet", num_classes=len(CLASSES), pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

# --- Ensemble (dinoV) Model Definition ---
class DinoHead(nn.Module):
    def __init__(self, in_dim=384, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def load_ensemble_model(checkpoint_path):
    # 1. DINOv2
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
    dinov2.eval()
    
    # 2. ConvNeXt
    convnext = models.convnext_tiny(pretrained=False)
    convnext.classifier[2] = nn.Linear(768, len(CLASSES))
    convnext = convnext.to(DEVICE).eval()
    
    # 3. Dino Head
    dino_head = DinoHead(384, len(CLASSES)).to(DEVICE).eval()
    
    # 4. Temp Scaler
    temp_scaler = TemperatureScaler().to(DEVICE).eval()
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    convnext.load_state_dict(ckpt['convnext'])
    dino_head.load_state_dict(ckpt['dino_head'])
    temp_scaler.load_state_dict(ckpt['temp_scaler'])
    
    return {
        "dinov2": dinov2,
        "convnext": convnext,
        "dino_head": dino_head,
        "temp_scaler": temp_scaler
    }

# --- Processing Functions ---

def download_youtube_video(url, output_path):
    base_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb', 'web_creator']
            }
        }
    }
    
    # 0. Check cache before downloading
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'noplaylist': True}) as ydl_check:
            info_check = ydl_check.extract_info(url, download=False)
            cached_filename = ydl_check.prepare_filename(info_check)
            if os.path.exists(cached_filename) and os.path.getsize(cached_filename) > 0:
                print(f"Using already downloaded video from cache: {cached_filename}")
                return cached_filename
    except Exception as e_cache:
        print(f"Cache check skipped ({e_cache}). Starting download...")

    # Try 1: Android / iOS player client (bypasses bot detection without login)
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e1:
        print(f"Direct download failed ({e1}), attempting browser cookies fallback...")

    # Try 2: Attempt using cookies from installed browsers
    browsers = ['chrome', 'edge', 'firefox', 'brave', 'opera']
    for b in browsers:
        try:
            print(f"Trying cookies from {b}...")
            opts = base_opts.copy()
            opts['cookiesfrombrowser'] = (b,)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info)
        except Exception:
            continue

    # Try 3: Last fallback with format='best' and android client
    opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android']
            }
        }
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def extract_frames(video_path, output_folder, fps=1):
    """
    Optimized frame extraction using FFmpeg. 
    It's multi-threaded and much faster than OpenCV.
    """
    import subprocess
    os.makedirs(output_folder, exist_ok=True)
    
    # Use ffmpeg for high-speed extraction
    # -hwaccel auto: use GPU acceleration if available
    # -q:v 2: high quality JPEGs
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    cmd = [
        'ffmpeg', '-hwaccel', 'auto', 
        '-i', video_path, 
        '-vf', f'fps={fps}', 
        '-q:v', '2', 
        output_pattern, 
        '-y'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        # Fallback to manual if needed, but ffmpeg is preferred
    
    # Get list of saved paths
    saved_paths = [str(p) for p in Path(output_folder).glob("frame_*.jpg")]
    saved_paths.sort()
    return saved_paths

from torch.utils.data import DataLoader, Dataset

class FrameDataset(Dataset):
    def __init__(self, frame_paths, transform):
        self.frame_paths = frame_paths
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path

def classify_frames_cnn(model, frame_paths, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FrameDataset(frame_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    results = []
    with torch.no_grad():
        for batch_imgs, paths in dataloader:
            batch_imgs = batch_imgs.to(DEVICE)
            logits = model(batch_imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for i in range(len(paths)):
                results.append({
                    "path": paths[i],
                    "label": CLASSES[preds[i].item()],
                    "confidence": probs[i][preds[i]].item()
                })
    return results

def classify_frames_clip(clip_model, classifier, preprocess, frame_paths, batch_size=32):
    dataset = FrameDataset(frame_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    results = []
    with torch.no_grad():
        for batch_imgs, paths in dataloader:
            batch_imgs = batch_imgs.to(DEVICE)
            
            features = clip_model.encode_image(batch_imgs)
            features = features / features.norm(dim=-1, keepdim=True)
            
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for i in range(len(paths)):
                results.append({
                    "path": paths[i],
                    "label": CLASSES[preds[i].item()],
                    "confidence": probs[i][preds[i]].item()
                })
    return results

def classify_frames_ensemble(models_dict, frame_paths, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FrameDataset(frame_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dinov2 = models_dict["dinov2"]
    convnext = models_dict["convnext"]
    dino_head = models_dict["dino_head"]
    temp_scaler = models_dict["temp_scaler"]
    
    results = []
    with torch.no_grad():
        for batch_imgs, paths in dataloader:
            batch_imgs = batch_imgs.to(DEVICE)
            
            # DINOv2 features
            dino_feat = dinov2.forward_features(batch_imgs)["x_norm_patchtokens"].mean(dim=1)
            dino_logits = dino_head(dino_feat)
            
            # ConvNeXt logits
            conv_logits = convnext(batch_imgs)
            
            # Ensemble & Calibrate
            logits = (dino_logits + conv_logits) / 2
            logits = temp_scaler(logits)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for i in range(len(paths)):
                results.append({
                    "path": paths[i],
                    "label": CLASSES[preds[i].item()],
                    "confidence": probs[i][preds[i]].item()
                })
    return results

# --- Audio Processing & Sensitive Word Muting ---

SENSITIVE_WORDS = [
    "fuck", "fucking", "fucked", "fucker", "shit", "shitty", "bullshit",
    "bitch", "bastard", "ass", "asshole", "dick", "cock", "piss", "crap",
    "damn", "idiot", "stupid", "dumb", "moron", "loser", "jerk", "retard",
    "sex", "porn", "nude", "naked", "boobs", "tits", "penis", "vagina",
    "horny", "blowjob", "kill", "murder", "shoot", "gun", "bomb", "explode",
    "stab", "knife", "suicide", "son of a bitch", "motherfucker", "piece of shit",
    "what the fuck", "what the hell", "go to hell", "damn it", "holy shit"
]

def process_video_audio(video_path, output_dir, mode="mute", log_fn=print):
    """
    Extracts audio from video, transcribes text with Whisper, finds curse/sensitive words,
    mutes/censors those words in audio, and exports cleaned audio.
    """
    import subprocess
    from pydub import AudioSegment

    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "extracted_audio.wav")
    
    # 1. Extract audio via ffmpeg
    log_fn("[20%] Extracting audio track from video using FFmpeg...")
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", wav_path, "-y"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log_fn("[35%] Audio track extracted successfully.")
    except Exception as e:
        log_fn(f"FFmpeg audio extraction note: {e}")
        if not os.path.exists(wav_path):
            return {"success": False, "reason": "No audio track found or extraction failed"}

    # 2. Transcribe using Whisper
    try:
        import whisper
        log_fn("[45%] Loading Whisper speech recognition model 'base'...")
        whisper_model = whisper.load_model("base")
        log_fn(f"[60%] Transcribing audio text: {os.path.basename(wav_path)}...")
        result = whisper_model.transcribe(wav_path, word_timestamps=True)
        log_fn("[75%] Speech transcription complete.")
    except Exception as e:
        log_fn(f"Whisper transcription error: {e}")
        return {"success": False, "reason": f"Whisper transcription error: {str(e)}"}

    full_text = result.get("text", "")
    flagged_segments = []
    found_words_info = []
    sensitive_words_lower = [w.lower() for w in SENSITIVE_WORDS]

    # 3. Identify curse/sensitive words
    log_fn("[80%] Scanning transcription text for sensitive & curse words...")
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word_text = word_info["word"].strip().strip(".,!?").lower()
            if word_text in sensitive_words_lower:
                start_sec = round(word_info["start"], 2)
                end_sec = round(word_info["end"], 2)
                start_ms = int(start_sec * 1000)
                end_ms = int(end_sec * 1000)
                flagged_segments.append((start_ms, end_ms))

                # Format in minutes as well (e.g., 216.16s -> 3m 36.16s)
                start_min = f"{int(start_sec // 60)}m {round(start_sec % 60, 2):05.2f}s" if start_sec >= 60 else f"{start_sec}s"
                end_min = f"{int(end_sec // 60)}m {round(end_sec % 60, 2):05.2f}s" if end_sec >= 60 else f"{end_sec}s"

                found_words_info.append({
                    "word": word_info["word"].strip(),
                    "start": start_sec,
                    "end": end_sec,
                    "start_fmt": start_min,
                    "end_fmt": end_min
                })

    log_fn(f"[88%] Found {len(found_words_info)} curse/sensitive word(s) in audio.")

    # 4. Mute flagged segments
    try:
        audio = AudioSegment.from_wav(wav_path)
        if flagged_segments:
            log_fn(f"[92%] Muting {len(flagged_segments)} sensitive segment(s)...")
            flagged_segments.sort(key=lambda x: x[0])
            cleaned_audio = audio
            for start_ms, end_ms in flagged_segments:
                duration = end_ms - start_ms
                if duration <= 0:
                    continue
                silence = AudioSegment.silent(duration=duration)
                cleaned_audio = cleaned_audio[:start_ms] + silence + cleaned_audio[end_ms:]
        else:
            cleaned_audio = audio

        cleaned_audio_path = os.path.join(output_dir, "cleaned_audio.mp3")
        cleaned_audio.export(cleaned_audio_path, format="mp3")
        log_fn(f"[100%] Cleaned audio exported to: {os.path.basename(cleaned_audio_path)}")
    except Exception as e:
        log_fn(f"Audio muting export note: {e}")
        cleaned_audio_path = wav_path

    return {
        "success": True,
        "transcription": full_text,
        "flagged_words": found_words_info,
        "flagged_count": len(found_words_info),
        "cleaned_audio_path": cleaned_audio_path
    }

def create_final_safe_video(video_path, raw_results, audio_path, output_dir, fps=1.0, log_fn=print):
    """
    Intelligently cuts out unsafe video scenes (violence/erotism) and merges the
    cleaned safe video segments with the muted audio track into final_safe_video.mp4.
    """
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    final_video_path = os.path.join(output_dir, "final_safe_video.mp4")

    # 1. Identify unsafe frame timestamps
    unsafe_labels = {"violence", "violent", "erotism", "erotic"}
    unsafe_times = []
    
    for idx, res in enumerate(raw_results):
        lbl = res.get("label", "").lower()
        if lbl in unsafe_labels:
            # Frame index to approximate time in seconds
            t_sec = idx / float(fps) if fps > 0 else idx
            unsafe_times.append(t_sec)

    log_fn(f"Detected {len(unsafe_times)} unsafe frame instance(s) out of {len(raw_results)} total frames.")

    # 2. If no unsafe video scenes, merge original video with cleaned audio directly
    if not unsafe_times:
        log_fn("No unsafe video scenes detected. Merging full video with cleaned audio...")
        cmd = [
            "ffmpeg", "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", final_video_path, "-y"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            log_fn(f"[100%] Final safe video synthesized: {os.path.basename(final_video_path)}")
            return final_video_path, []
        except Exception as e:
            log_fn(f"Direct video/audio merge note: {e}")
            return video_path, []

    # 3. Build contiguous unsafe time windows [start, end]
    interval_buffer = 1.5 / float(fps) if fps > 0 else 1.5
    unsafe_windows = []
    for t in unsafe_times:
        w_start = max(0.0, t - 0.5)
        w_end = t + interval_buffer
        if not unsafe_windows:
            unsafe_windows.append([w_start, w_end])
        else:
            prev_start, prev_end = unsafe_windows[-1]
            if w_start <= prev_end + 1.0: # Merge close scenes
                unsafe_windows[-1][1] = max(prev_end, w_end)
            else:
                unsafe_windows.append([w_start, w_end])

    # Format cut scene metadata for UI tracking
    cut_scenes_metadata = []
    for idx_win, (w_start, w_end) in enumerate(unsafe_windows):
        s_sec = round(w_start, 2)
        e_sec = round(w_end, 2)
        dur_sec = round(e_sec - s_sec, 2)
        
        s_min = f"{int(s_sec // 60)}m {round(s_sec % 60, 2):05.2f}s" if s_sec >= 60 else f"{s_sec}s"
        e_min = f"{int(e_sec // 60)}m {round(e_sec % 60, 2):05.2f}s" if e_sec >= 60 else f"{e_sec}s"
        d_min = f"{int(dur_sec // 60)}m {round(dur_sec % 60, 2):05.2f}s" if dur_sec >= 60 else f"{dur_sec}s"

        cut_scenes_metadata.append({
            "index": idx_win + 1,
            "start": s_sec,
            "end": e_sec,
            "start_fmt": s_min,
            "end_fmt": e_min,
            "duration_sec": dur_sec,
            "duration_fmt": d_min,
            "reason": "Unsafe Content (Violence / Erotism)"
        })

    log_fn(f"Built {len(cut_scenes_metadata)} unsafe scene cut window(s): {unsafe_windows}")

    # 4. Step 1: Attach Safe Muted Audio to Original Video First
    synced_full_video = os.path.join(output_dir, "synced_full_video.mp4")
    log_fn("[92%] Step 1: Attaching safe/muted audio to full video before surgical cutting...")
    
    merge_cmd = [
        "ffmpeg", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", synced_full_video, "-y"
    ]
    try:
        subprocess.run(merge_cmd, check=True, capture_output=True)
    except Exception as e_merge:
        log_fn(f"Warning attaching safe audio: {e_merge}")
        synced_full_video = video_path

    # 5. Get video duration via ffprobe
    duration = 0.0
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", synced_full_video
        ]
        probe_out = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        duration = float(probe_out.stdout.strip())
    except Exception:
        duration = (len(raw_results) / float(fps)) + 2.0 if fps > 0 else 300.0

    # 6. Compute Safe Complement Windows [start, end]
    safe_windows = []
    curr_time = 0.0
    for u_start, u_end in unsafe_windows:
        if u_start > curr_time + 0.3:
            safe_windows.append([curr_time, u_start])
        curr_time = max(curr_time, u_end)
    if curr_time < duration - 0.3:
        safe_windows.append([curr_time, duration])

    log_fn(f"[95%] Step 2: Surgical Cut Method - Extracting {len(safe_windows)} joint (Video+Audio) safe segment(s)...")

    # 7. Extract safe sub-segments with BOTH Video + Audio
    temp_segments = []
    concat_list_path = os.path.join(output_dir, "concat_list.txt")

    try:
        for idx, (s_start, s_end) in enumerate(safe_windows):
            seg_file = os.path.join(output_dir, f"safe_seg_{idx:03d}.mp4")
            seg_cmd = [
                "ffmpeg", "-ss", str(round(s_start, 2)), "-to", str(round(s_end, 2)),
                "-i", synced_full_video,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-avoid_negative_ts", "make_zero",
                seg_file, "-y"
            ]
            subprocess.run(seg_cmd, check=True, capture_output=True)
            temp_segments.append(seg_file)

        # Write concat list file
        with open(concat_list_path, "w") as f:
            for seg in temp_segments:
                seg_esc = seg.replace("\\", "/")
                f.write(f"file '{seg_esc}'\n")

        # 8. Concatenate joint safe (Video + Audio) segments together for 100% sync
        log_fn("[98%] Step 3: Re-stitching joint safe segments into final synchronized video...")
        cat_cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c:v", "copy", "-c:a", "copy", final_video_path, "-y"
        ]
        subprocess.run(cat_cmd, check=True, capture_output=True)

        log_fn(f"[100%] Surgical Cut Complete! Final safe synced video synthesized: {os.path.basename(final_video_path)}")
        return final_video_path, cut_scenes_metadata

    except Exception as e_cut:
        log_fn(f"Surgical cut fallback note: {e_cut}")
        return synced_full_video, cut_scenes_metadata
