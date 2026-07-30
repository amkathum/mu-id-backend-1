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
JOB_TTL = 7200                   # seconds before a completed job is purged (2 hours)

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
    """Create a new job entry and return its ID. Also purges expired jobs inline."""
    job_id = uuid.uuid4().hex
    cutoff = time.time() - JOB_TTL
    with jobs_lock:
        # Inline cleanup: remove jobs older than JOB_TTL
        expired = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
        for jid in expired:
            del jobs[jid]
        jobs[job_id] = {
            "status": "processing",
            "progress": "",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    return job_id


def _finish_job(job_id: str, result: dict) -> None:
    with jobs_lock:
        jobs[job_id].update({"status": "completed", "result": result})


def _fail_job(job_id: str, message: str) -> None:
    with jobs_lock:
        jobs[job_id].update({"status": "failed", "error": message})


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
# Startup checks
# ---------------------------------------------------------------------------
def _check_dependencies() -> None:
    """Log ffmpeg availability at startup so missing binaries are caught early."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        first_line = r.stdout.splitlines()[0] if r.stdout else "(no output)"
        print(f"[startup] ffmpeg found: {first_line}")
    except FileNotFoundError:
        print("[startup] WARNING: ffmpeg not found — compression step will fail")

_check_dependencies()

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
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(src),
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            str(dst),
            "-y",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ffmpeg] stderr:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with status {result.returncode}:\n{result.stderr}")


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
                    jobs[job_id]["progress"] = f"transcribing chunk {i + 1}/{total}"
        parts.append(transcribe_file(chunk, client))
    return "\n".join(parts)


def _llama_call(client: Groq, messages: list[dict]) -> str:
    """Single LLaMA call with a hard 120-second timeout."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=8192,
        timeout=120,
    )
    return response.choices[0].message.content or ""


def _parse_latex_response(content: str) -> tuple[str, str]:
    """Extract subject and latex body from a LLaMA response."""
    subject = ""
    latex = ""
    if "SUBJECT:" in content and "LATEX:" in content:
        subject_part, latex_part = content.split("LATEX:", 1)
        subject = subject_part.replace("SUBJECT:", "").strip()
        latex = latex_part.strip()
    else:
        latex = content
    return subject, latex


def _extract_body(latex_doc: str) -> str:
    """Pull the content between \\begin{document} and \\end{document}."""
    begin = latex_doc.find(r"\begin{document}")
    end = latex_doc.find(r"\end{document}")
    if begin != -1 and end != -1:
        return latex_doc[begin + len(r"\begin{document}"): end].strip()
    return latex_doc.strip()


def to_latex(transcription: str, client: Groq) -> dict:
    """Convert transcription to LaTeX.

    If the transcript exceeds MAX_LATEX_CHARS it is split into two halves,
    each converted separately, then merged into one document.
    """
    print(f"[latex] transcript length={len(transcription)}, MAX_LATEX_CHARS={MAX_LATEX_CHARS}")

    if len(transcription) <= MAX_LATEX_CHARS:
        # ── Single-pass (short transcript) ──────────────────────────────────
        print("[latex] starting single-pass conversion...")
        try:
            content = _llama_call(client, [
                {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcription:\n\n{transcription}"},
            ])
        except Exception as e:
            print(f"[latex] error on single-pass: {e}")
            raise
        subject, latex = _parse_latex_response(content)
        print(f"[latex] done (single-pass), length={len(latex)}")
        return {"subject": subject, "latex": latex}

    # ── Two-pass (long transcript) ───────────────────────────────────────────
    mid = len(transcription) // 2
    part1 = transcription[:mid]
    part2 = transcription[mid:]
    print(f"[latex] long transcript — splitting into 2 parts ({len(part1)}, {len(part2)} chars)")

    # Part 1 — full document + subject
    print("[latex] starting part-1 conversion...")
    try:
        content1 = _llama_call(client, [
            {"role": "system", "content": LATEX_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcription (part 1 of 2):\n\n{part1}"},
        ])
    except Exception as e:
        print(f"[latex] error on part-1: {e}")
        raise
    subject, latex1 = _parse_latex_response(content1)
    print(f"[latex] part-1 done, length={len(latex1)}")

    # Part 2 — body content only (no preamble needed)
    part2_prompt = (
        "Continue converting the following Arabic transcription (part 2 of 2) into "
        "XeLaTeX sections. Return ONLY the LaTeX body content — no preamble, no "
        "\\begin{document}, no \\end{document}. Just \\section/\\subsection/\\begin{itemize} etc."
    )
    print("[latex] starting part-2 conversion...")
    try:
        latex2_body = _llama_call(client, [
            {"role": "system", "content": LATEX_SYSTEM_PROMPT},
            {"role": "user", "content": f"{part2_prompt}\n\n{part2}"},
        ])
    except Exception as e:
        print(f"[latex] error on part-2: {e}")
        raise
    print(f"[latex] part-2 done, length={len(latex2_body)}")

    # Merge: inject part-2 body before \end{document} in the part-1 document
    body1 = _extract_body(latex1)
    end_tag = r"\end{document}"
    if end_tag in latex1:
        merged = latex1.replace(end_tag, f"\n{latex2_body.strip()}\n{end_tag}", 1)
    else:
        merged = latex1 + "\n" + latex2_body.strip()

    print(f"[latex] merge done, total length={len(merged)}")
    return {"subject": subject, "latex": merged}


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

    def _set_progress(msg: str) -> None:
        if job_id:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["progress"] = msg

    print(f"[pipeline] step 1/4 — compressing audio: {audio_path.name}")
    _set_progress("compressing audio")
    try:
        compressed = tmp_dir / "compressed.mp3"
        compress_audio(audio_path, compressed)
        print(f"[pipeline] step 1/4 — done, size={compressed.stat().st_size} bytes")
    except Exception as e:
        print(f"[pipeline] step 1/4 — compress failed: {e}")
        raise

    print("[pipeline] step 2/4 — chunking audio")
    _set_progress("splitting audio into chunks")
    try:
        chunks = chunk_audio(compressed, tmp_dir)
        print(f"[pipeline] step 2/4 — done, {len(chunks)} chunk(s)")
    except Exception as e:
        print(f"[pipeline] step 2/4 — chunking failed: {e}")
        raise

    print(f"[pipeline] step 3/4 — transcribing {len(chunks)} chunk(s)")
    _set_progress(f"transcribing chunk 1/{len(chunks)}")
    try:
        transcription = transcribe_chunks(chunks, client, job_id=job_id)
        print(f"[pipeline] step 3/4 — done, transcript length={len(transcription)}")
    except Exception as e:
        print(f"[pipeline] step 3/4 — transcription failed: {e}")
        raise

    print("[pipeline] step 4/4 — converting to LaTeX")
    _set_progress("converting to LaTeX")
    try:
        latex_result = to_latex(transcription, client)
        print("[pipeline] step 4/4 — done")
    except Exception as e:
        print(f"[pipeline] step 4/4 — LaTeX conversion failed: {e}")
        raise

    return {
        "transcript": transcription,
        "subject": latex_result["subject"],
        "latex": latex_result["latex"],
    }


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def _run_youtube_job(job_id: str, youtube_url: str, max_duration: int = 7200) -> None:
    tmp_dir = make_tmp()
    try:
        duration = get_video_duration(youtube_url)
        if duration > max_duration:
            hours = max_duration // 3600
            _fail_job(job_id, f"الفيديو طويل جداً، الحد الأقصى {hours} ساعات حالياً")
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


@app.route("/api/transcribe-async", methods=["POST"])
def transcribe_youtube_async():
    """Long-video async transcription — supports up to 6 hours. Returns job_id immediately.

    Accepts either:
    - JSON body:              {"youtube_url": "..."}
    - multipart/form-data:   file field named 'file'
    """
    # ── File upload path ────────────────────────────────────────────────────
    if request.content_type and "multipart/form-data" in request.content_type:
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

    # ── YouTube URL path ────────────────────────────────────────────────────
    data = request.get_json(silent=True) or {}
    youtube_url = data.get("youtube_url", "").strip()

    if not youtube_url:
        return jsonify({"error": "Provide either a youtube_url (JSON) or a file (multipart)"}), 400

    job_id = _new_job()
    thread = threading.Thread(
        target=_run_youtube_job,
        args=(job_id, youtube_url, 21600),  # 6-hour limit
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

    if status == "completed":
        result = job.get("result") or {}
        return jsonify({
            "status": "completed",
            "transcript": result.get("transcript", ""),
            "subject": result.get("subject", ""),
            "latex": result.get("latex", ""),
        })

    # status == "failed"
    return jsonify({"status": "failed", "error": job.get("error", "Unknown error")})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
