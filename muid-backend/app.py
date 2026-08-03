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
import requests
import yt_dlp
from pydub import AudioSegment

app = Flask(__name__)
CORS(app, origins="*")
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MAX_CHUNK_MS = 8 * 60 * 1000   # 8 minutes in milliseconds
MAX_LATEX_CHARS = 2000           # max chars sent to LLaMA per call
JOB_TTL = 7200                   # seconds before a completed job is purged (2 hours)

LATEX_SYSTEM_PROMPT = """\
You are an expert Arabic academic typesetter.
Given a transcription of an Arabic lecture or document, convert it into 
well-structured LaTeX BODY content only.

Return your response in EXACTLY this format (nothing else before or after):

SUBJECT: [English subject of the content]
LATEX: [LaTeX body content only — sections, subsections, tcolorboxes, itemize]

CRITICAL RULES:
- Do NOT include \\documentclass, \\usepackage, \\begin{document}, or \\end{document}
- Only output the BODY: \\section, \\subsection, \\begin{itemize}, \\begin{tcolorbox}, etc.
- Cover EVERY topic and detail from the transcription — do not summarize or skip content
- Each distinct topic must have its own unique \\section or \\subsection — 
  NEVER repeat the same heading title twice
- Use \\begin{tcolorbox} for definitions and key explanations
- Use \\begin{itemize} for lists of related points
"""

LATEX_PREAMBLE = r"""\documentclass{article}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{polyglossia}
\setmainlanguage{arabic}
\setotherlanguage{english}
\usepackage{fontspec}
\setmainfont{Amiri}
\usepackage{tcolorbox}
\usepackage{xcolor}
\begin{document}
"""

LATEX_POSTAMBLE = r"""
\end{document}
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
        max_tokens=4000,
        timeout=120,
    )
    time.sleep(15)  # respect Groq's 12,000 TPM limit
    return response.choices[0].message.content or ""


def _openrouter_call(messages: list[dict], model: str = "deepseek/deepseek-v4-flash:free") -> str:
    """Fallback call to OpenRouter when Groq hits a rate limit."""
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4000,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"] or ""


def _llama_call_with_fallback(client: Groq, messages: list[dict]) -> str:
    """Try Groq first; fall back to OpenRouter free models if Groq's rate limit is hit."""
    try:
        return _llama_call(client, messages)
    except Exception as e:
        error_str = str(e)
        if "rate_limit" in error_str or "413" in error_str or "429" in error_str:
            print(f"[fallback] Groq limit hit ({error_str[:100]}), trying OpenRouter...")
            try:
                return _openrouter_call(messages, model="deepseek/deepseek-v4-flash:free")
            except Exception as e2:
                print(f"[fallback] OpenRouter deepseek failed ({str(e2)[:100]}), trying gpt-oss...")
                return _openrouter_call(messages, model="openai/gpt-oss-120b:free")
        raise

def _parse_latex_response(content: str) -> tuple[str, str]:
    """Extract subject and latex body from a LLaMA response."""
    subject = ""
    latex = ""
    if "SUBJECT:" in content and "LATEX:" in content:
        subject_part, latex_part = content.split("LATEX:", 1)
        subject = subject_part.replace("SUBJECT:", "").strip()
        latex = latex_part.strip()
    else:
        print(f"[latex-parse] WARNING: response missing SUBJECT:/LATEX: markers. First 200 chars: {content[:200]}")
            latex = ""  # empty instead of raw transcript — signals failure clearly
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

        Splits the transcript into small chunks (each under MAX_LATEX_CHARS)
        so LLaMA can cover every detail without being forced to summarize.
        """
        print(f"[latex] transcript length={len(transcription)}, MAX_LATEX_CHARS={MAX_LATEX_CHARS}")

        parts = []
        remaining = transcription
        while remaining:
            if len(remaining) <= MAX_LATEX_CHARS:
                parts.append(remaining)
                break
            cut = remaining.rfind(".", 0, MAX_LATEX_CHARS)
            if cut == -1 or cut < MAX_LATEX_CHARS // 2:
                cut = MAX_LATEX_CHARS
            parts.append(remaining[:cut])
            remaining = remaining[cut:]

        print(f"[latex] split into {len(parts)} part(s) for full-detail conversion")

        subject = ""
        body_sections = []

        for i, part in enumerate(parts):
            is_first = (i == 0)
            part_label = f"part {i + 1} of {len(parts)}"
            print(f"[latex] starting {part_label} conversion...")

            if is_first:
                user_content = f"Transcription ({part_label}):\n\n{part}"
            else:
                user_content = (
                    f"Continue converting the following Arabic transcription ({part_label}) "
                    f"into LaTeX body content. Return ONLY the LaTeX body — no SUBJECT line needed "
                    f"since it was already provided in part 1.\n\n{part}"
                )

            try:
                content = _llama_call_with_fallback(client, [
                    {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ])
            except Exception as e:
                print(f"[latex] error on {part_label}: {e}")
                raise

            part_subject, part_body = _parse_latex_response(content)

            if r"\section" not in part_body and r"\subsection" not in part_body:
                print(f"[latex] WARNING: {part_label} doesn't look like LaTeX, retrying once. First 200 chars: {part_body[:200]}")
                try:
                    content = _llama_call_with_fallback(client, [
                        {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ])
                    part_subject, part_body = _parse_latex_response(content)
                except Exception as e:
                    print(f"[latex] retry on {part_label} also failed: {e}")

            if is_first and part_subject:
                subject = part_subject

            body_sections.append(part_body)
            print(f"[latex] {part_label} done, length={len(part_body)}")

        full_body = "\n\n".join(body_sections)
        full_latex = LATEX_PREAMBLE + full_body + LATEX_POSTAMBLE
        print(f"[latex] all parts merged, total length={len(full_latex)}")
        return {"subject": subject, "latex": full_latex}
    mid = len(transcription) // 2
    part1 = transcription[:mid]
    part2 = transcription[mid:]
    print(f"[latex] long transcript — splitting into 2 parts ({len(part1)}, {len(part2)} chars)")

    print("[latex] starting part-1 conversion...")
    try:
        content1 = _llama_call_with_fallback(client, [
            {"role": "system", "content": LATEX_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcription (part 1 of 2):\n\n{part1}"},
        ])
    except Exception as e:
        print(f"[latex] error on part-1: {e}")
        raise
    subject, latex1 = _parse_latex_response(content1)
    print(f"[latex] part-1 done, length={len(latex1)}")

    part2_prompt = (
        "Continue converting the following Arabic transcription (part 2 of 2) into "
        "XeLaTeX sections. Return ONLY the LaTeX body content — no preamble, no "
        "\\begin{document}, no \\end{document}. Just \\section/\\subsection/\\begin{itemize} etc."
    )
    print("[latex] starting part-2 conversion...")
    try:
        latex2_body = _llama_call_with_fallback(client, [
            {"role": "system", "content": LATEX_SYSTEM_PROMPT},
            {"role": "user", "content": f"{part2_prompt}\n\n{part2}"},
        ])
    except Exception as e:
        print(f"[latex] error on part-2: {e}")
        raise
    # Retry once if the response doesn't look like LaTeX (missing key markers)
    if r"\section" not in latex2_body and r"\subsection" not in latex2_body:
        print(f"[latex] WARNING: part-2 doesn't look like LaTeX, retrying once. First 200 chars: {latex2_body[:200]}")
        try:
            latex2_body = _llama_call_with_fallback(client, [
                {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                {"role": "user", "content": f"{part2_prompt}\n\n{part2}"},
            ])
        except Exception as e:
            print(f"[latex] retry on part-2 also failed: {e}")
    print(f"[latex] part-2 done, length={len(latex2_body)}")

    end_tag = r"\end{document}"
    if end_tag in latex1:
        merged = latex1.replace(end_tag, f"\n{latex2_body.strip()}\n{end_tag}", 1)
    else:
        merged = latex1 + "\n" + latex2_body.strip()

    full_latex = LATEX_PREAMBLE + merged + LATEX_POSTAMBLE
    print(f"[latex] merge done, total length={len(full_latex)}")
    return {"subject": subject, "latex": full_latex}


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
    print(f"[transcribe-async] content_type={request.content_type}")
    print(f"[transcribe-async] files={list(request.files.keys())}")
    print(f"[transcribe-async] form={dict(request.form)}")
    if request.content_type and "multipart/form-data" in request.content_type:
        uploaded = None
        for field in ["file", "audio", "upload", "video"]:
            if field in request.files:
                uploaded = request.files[field]
                break
        if uploaded is None:
            return jsonify({"error": "No file found. Use field name: file, audio, upload, or video"}), 400
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
    uploaded = None
    for field in ["file", "audio", "upload", "video"]:
        if field in request.files:
            uploaded = request.files[field]
            break
    if uploaded is None:
        return jsonify({"error": "No file found. Use field name: file, audio, upload, or video"}), 400
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

    return jsonify({"status": "failed", "error": job.get("error", "Unknown error")})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)