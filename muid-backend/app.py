import os
import uuid
import shutil
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import yt_dlp
from pydub import AudioSegment

app = Flask(__name__)
CORS(app, origins="*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MAX_CHUNK_MS = 8 * 60 * 1000   # 8 minutes in milliseconds
MAX_LATEX_CHARS = 8000           # max chars sent to LLaMA per call
JOB_TTL = 3600                   # seconds before a completed job is purged (1 hour)

LATEX_SYSTEM_PROMPT = """\
You are an expert Arabic academic typesetter.
Given a transcription of an Arabic lecture or document, convert it into a
beautiful, well-structured XeLaTeX document.

Return your response in EXACTLY this format (nothing else before or after):

SUBJECT: [English subject of the content]
LATEX: [complete, compilable XeLaTeX source code]

LaTeX requirements:
- \\documentclass{article}
- Use Polyglossia: \\setmainlanguage{arabic}, \\setotherlanguage{english}
- Use Amiri font: \\setmainfont{Amiri}
- RTL layout: \\usepackage{geometry} with appropriate margins
- Use \\usepackage{tcolorbox} for highlighted boxes
- Organize content with \\section, \\subsection, \\begin{itemize} etc.
- Include \\usepackage{xcolor} for colours
- The document must be self-contained and compile with xelatex
- Do NOT truncate the output; include every detail from the transcription
"""

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _new_job() -> str:
    """Create a new job entry and return its ID."""
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "status": "processing",
            "created_at": time.time(),
        }
    return job_id


def _finish_job(job_id: str, result: dict) -> None:
    with jobs_lock:
        jobs[job_id].update({"status": "done", **result})


def _fail_job(job_id: str, message: str) -> None:
    with jobs_lock:
        jobs[job_id].update({"status": "error", "message": message})


def _cleanup_loop() -> None:
    """Background thread: purge jobs older than JOB_TTL every 10 minutes."""
    while True:
        time.sleep(600)
        cutoff = time.time() - JOB_TTL
        with jobs_lock:
            expired = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
            for jid in expired:
                del jobs[jid]


# Start the cleanup daemon immediately
_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()

# ---------------------------------------------------------------------------
# Audio / transcription helpers
# ---------------------------------------------------------------------------


def make_tmp() -> Path:
    """Create and return a unique temp directory under /tmp."""
    tmp = Path(f"/tmp/muid_{uuid.uuid4().hex}")
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def compress_audio(src: Path, dst: Path) -> None:
    """Downsample to 16 kHz mono 32 kbps MP3 for smaller Groq payloads."""
    subprocess.run(
        [
            "ffmpeg", "-i", str(src),
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            str(dst),
            "-y",
        ],
        check=True,
        capture_output=True,
    )


def chunk_audio(audio_path: Path, tmp_dir: Path) -> list[Path]:
    """Split audio into 8-minute chunks; returns list of chunk paths."""
    audio = AudioSegment.from_file(str(audio_path))
    chunks = []
    for i, start_ms in enumerate(range(0, len(audio), MAX_CHUNK_MS)):
        chunk = audio[start_ms: start_ms + MAX_CHUNK_MS]
        chunk_path = tmp_dir / f"chunk_{i:04d}.mp3"
        chunk.export(str(chunk_path), format="mp3")
        chunks.append(chunk_path)
    return chunks


def transcribe_file(audio_path: Path, client: Groq) -> str:
    """Transcribe a single audio file using Groq Whisper."""
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language="ar",
            response_format="text",
        )
    return result if isinstance(result, str) else result.text


def transcribe_chunks(chunks: list[Path], client: Groq, job_id: str | None = None) -> str:
    """Transcribe each chunk and join results, updating job progress after each one."""
    total = len(chunks)
    parts = []
    for i, chunk in enumerate(chunks):
        if job_id:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["progress"] = f"جاري تفريغ الجزء {i + 1} من {total}"
        parts.append(transcribe_file(chunk, client))
    return "\n".join(parts)


def to_latex(transcription: str, client: Groq) -> dict:
    """Convert transcription text to LaTeX via Groq LLaMA."""
    text = transcription[:MAX_LATEX_CHARS] if len(transcription) > MAX_LATEX_CHARS else transcription

    print("[latex] starting conversion...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcription:\n\n{text}"},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
    except Exception as e:
        print(f"[latex] error: {e}")
        raise

    content = response.choices[0].message.content or ""
    subject = ""
    latex = ""
    if "SUBJECT:" in content and "LATEX:" in content:
        subject_part, latex_part = content.split("LATEX:", 1)
        subject = subject_part.replace("SUBJECT:", "").strip()
        latex = latex_part.strip()
    else:
        latex = content

    print(f"[latex] done, length={len(latex)}")
    return {"subject": subject, "latex": latex}


def get_video_duration(url: str) -> float:
    """Return video duration in seconds without downloading the file."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "nocheckcertificate": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["ios"],
                "player_skip": ["webpage", "configs"],
            }
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return float(info.get("duration") or 0)


def download_youtube_audio(url: str, tmp_dir: Path) -> Path:
    """Download best-audio from a YouTube URL, extract to MP3."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "extractor_args": {
            "youtube": {
                "player_client": ["ios"],
                "player_skip": ["webpage", "configs"],
            }
        },
        "nocheckcertificate": True,
        "no_warnings": False,
        "ignoreerrors": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    mp3_files = list(tmp_dir.glob("audio.mp3"))
    if not mp3_files:
        candidates = list(tmp_dir.iterdir())
        if not candidates:
            raise FileNotFoundError("yt-dlp did not produce an output file.")
        return candidates[0]
    return mp3_files[0]


def process_audio(audio_path: Path, tmp_dir: Path, job_id: str | None = None) -> dict:
    """Core pipeline: compress → chunk → transcribe → latex."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    client = Groq(api_key=GROQ_API_KEY)

    compressed = tmp_dir / "compressed.mp3"
    compress_audio(audio_path, compressed)

    chunks = chunk_audio(compressed, tmp_dir)
    transcription = transcribe_chunks(chunks, client, job_id=job_id)
    latex_result = to_latex(transcription, client)

    return {
        "transcript": transcription,
        "subject": latex_result["subject"],
        "latex": latex_result["latex"],
    }


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def _run_youtube_job(job_id: str, youtube_url: str) -> None:
    tmp_dir = make_tmp()
    try:
        duration = get_video_duration(youtube_url)
        if duration > 7200:
            _fail_job(job_id, "الفيديو طويل جداً، الحد الأقصى ساعتان حالياً")
            return
        audio_path = download_youtube_audio(youtube_url, tmp_dir)
        result = process_audio(audio_path, tmp_dir, job_id=job_id)
        _finish_job(job_id, result)
    except Exception as exc:
        _fail_job(job_id, str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_upload_job(job_id: str, audio_path: Path, tmp_dir: Path) -> None:
    try:
        result = process_audio(audio_path, tmp_dir, job_id=job_id)
        _finish_job(job_id, result)
    except Exception as exc:
        _fail_job(job_id, str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/transcribe", methods=["POST"])
def transcribe_youtube():
    """Kick off a background YouTube transcription job; return job_id immediately."""
    data = request.get_json(silent=True) or {}
    youtube_url = data.get("youtube_url", "").strip()

    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    job_id = _new_job()
    thread = threading.Thread(
        target=_run_youtube_job,
        args=(job_id, youtube_url),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "processing"}), 202


@app.route("/api/upload", methods=["POST"])
def upload_audio():
    """Kick off a background upload transcription job; return job_id immediately."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    tmp_dir = make_tmp()
    suffix = Path(uploaded.filename).suffix or ".mp3"
    audio_path = tmp_dir / f"upload{suffix}"
    uploaded.save(str(audio_path))

    job_id = _new_job()
    thread = threading.Thread(
        target=_run_upload_job,
        args=(job_id, audio_path, tmp_dir),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "processing"}), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Poll for job results."""
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found"}), 404

    status = job["status"]

    if status == "processing":
        return jsonify({
            "status": "processing",
            "progress": job.get("progress", ""),
        })

    if status == "done":
        return jsonify({
            "status": "done",
            "transcript": job.get("transcript", ""),
            "subject": job.get("subject", ""),
            "latex": job.get("latex", ""),
        })

    # status == "error"
    return jsonify({"status": "error", "message": job.get("message", "Unknown error")})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
