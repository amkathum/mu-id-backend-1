import os
import uuid
import shutil
import subprocess
import threading
import time
import math
import random
import requests
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
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

    MAX_CHUNK_MS = 8 * 60 * 1000   # 8 minutes in milliseconds
    MAX_LATEX_CHARS = 2000         # ↓ reduced from 3000 for lower token usage
    JOB_TTL = 7200                 # seconds before a completed job is purged (2 hours)

    # Models
    GROQ_LATEX_MODEL = "llama-3.1-8b-instant"      # 500K TPD (5× more than 70B)
    WHISPER_MODEL = "whisper-large-v3-turbo"       # cheaper & faster
    OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    LATEX_MAX_TOKENS = 4096                        # ↓ from 8192

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


    def _new_job(webhook_url: str = "") -> str:
        """Create a new job entry and return its ID. Also purges expired jobs inline."""
        job_id = uuid.uuid4().hex
        cutoff = time.time() - JOB_TTL
        with jobs_lock:
            expired = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
            for jid in expired:
                try:
                    del jobs[jid]
                except KeyError:
                    pass
            jobs[job_id] = {
                "status": "processing",
                "progress": "",
                "result": None,
                "error": None,
                "created_at": time.time(),
                "webhook_url": webhook_url,
            }
        return job_id


    def _finish_job(job_id: str, result: dict) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].update({"status": "completed", "result": result})


    def _fail_job(job_id: str, message: str) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].update({"status": "failed", "error": message})


    def _send_webhook(job_id: str) -> None:
        """Send webhook notification when job completes or fails."""
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            webhook_url = job.get("webhook_url", "")
            if not webhook_url:
                return

            status = job["status"]
            payload = {"job_id": job_id, "status": status}

            if status == "completed":
                result = job.get("result") or {}
                payload.update({
                    "transcript": result.get("transcript", ""),
                    "subject": result.get("subject", ""),
                    "latex": result.get("latex", ""),
                })
            elif status == "failed":
                payload["error"] = job.get("error", "Unknown error")
            else:
                return

        def _do_send():
            try:
                print(f"[webhook] Sending to {webhook_url} for job {job_id}")
                resp = requests.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                print(f"[webhook] Response: {resp.status_code}")
            except Exception as e:
                print(f"[webhook] Failed to send: {e}")

        threading.Thread(target=_do_send, daemon=True).start()


    def _cleanup_loop() -> None:
        """Background thread: purge jobs older than JOB_TTL every 10 minutes."""
        while True:
            time.sleep(600)
            cutoff = time.time() - JOB_TTL
            with jobs_lock:
                expired = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
                for jid in expired:
                    try:
                        del jobs[jid]
                    except KeyError:
                        pass


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

        if GROQ_API_KEY:
            print("[startup] GROQ_API_KEY found")
        else:
            print("[startup] WARNING: GROQ_API_KEY not set")

        if OPENROUTER_API_KEY:
            print("[startup] OPENROUTER_API_KEY found (fallback ready)")
        else:
            print("[startup] WARNING: OPENROUTER_API_KEY not set — fallback disabled")

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
                model=WHISPER_MODEL,
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


    # ---------------------------------------------------------------------------
    # LLM Call with Groq + OpenRouter Fallback + Retry
    # ---------------------------------------------------------------------------

    def _call_openrouter(messages: list[dict], max_tokens: int = 4096) -> str:
        """Call OpenRouter API as fallback."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OpenRouter API key not configured")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://replit.com",
            "X-Title": "Arabic Lecture LaTeX Converter",
        }

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }

        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        return data["choices"][0]["message"]["content"] or ""


    def _llama_call_with_fallback(messages: list[dict], max_retries: int = 5) -> str:
        """
        Try Groq first with retry + exponential backoff.
        If Groq fails completely, fallback to OpenRouter.
        """
        client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

        # ── Attempt 1: Groq with retry ──────────────────────────────────────
        if client:
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=GROQ_LATEX_MODEL,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=LATEX_MAX_TOKENS,
                        timeout=120,
                    )
                    print(f"[llm] Groq success on attempt {attempt + 1}")
                    return response.choices[0].message.content or ""
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in str(e) or "rate_limit" in err_str

                    if attempt < max_retries - 1:
                        wait = (2 ** attempt) + random.uniform(0, 2)
                        reason = "rate limit" if is_rate_limit else "error"
                        print(f"[llm] Groq {reason} attempt {attempt + 1}/{max_retries}. Waiting {wait:.1f}s...")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[llm] Groq failed after {max_retries} attempts. Falling back to OpenRouter...")
                        break
        else:
            print("[llm] GROQ_API_KEY not set, skipping Groq...")

        # ── Attempt 2: OpenRouter Fallback ────────────────────────────────
        if OPENROUTER_API_KEY:
            try:
                print(f"[llm] Trying OpenRouter with model {OPENROUTER_MODEL}...")
                content = _call_openrouter(messages, max_tokens=LATEX_MAX_TOKENS)
                print("[llm] OpenRouter success!")
                return content
            except Exception as e:
                print(f"[llm] OpenRouter also failed: {e}")
                raise RuntimeError(f"Both Groq and OpenRouter failed. Last error: {e}")
        else:
            raise RuntimeError("Groq failed and OpenRouter fallback is not configured.")


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


    def to_latex(transcription: str) -> dict:
        """Convert transcription to LaTeX with fallback support."""
        print(f"[latex] transcript length={len(transcription)}, MAX_LATEX_CHARS={MAX_LATEX_CHARS}")

        if len(transcription) <= MAX_LATEX_CHARS:
            # ── Single-pass (short transcript) ──────────────────────────────────
            print("[latex] starting single-pass conversion...")
            try:
                content = _llama_call_with_fallback([
                    {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Transcription:\n\n{transcription}"},
                ])
            except Exception as e:
                print(f"[latex] error on single-pass: {e}")
                raise
            subject, latex = _parse_latex_response(content)
            print(f"[latex] done (single-pass), length={len(latex)}")
            return {"subject": subject, "latex": latex}

        # ── Multi-pass (long transcript) ─────────────────────────────────────────
        num_parts = math.ceil(len(transcription) / MAX_LATEX_CHARS)
        part_size = math.ceil(len(transcription) / num_parts)
        parts = [transcription[i:i + part_size] for i in range(0, len(transcription), part_size)]
        print(f"[latex] long transcript — splitting into {len(parts)} parts of ~{part_size} chars each")

        subject = ""
        full_doc = None
        merged_body_parts = []

        for idx, part in enumerate(parts):
            is_first = (idx == 0)
            if is_first:
                prompt_content = f"Transcription (part {idx + 1} of {len(parts)}):\n\n{part}"
            else:
                prompt_content = (
                    "Continue converting the following Arabic transcription "
                    f"(part {idx + 1} of {len(parts)}) into XeLaTeX sections. Return ONLY "
                    "the LaTeX body content — no preamble, no \\begin{document}, no "
                    "\\end{document}. Just \\section/\\subsection/\\begin{itemize} etc.\n\n"
                    f"{part}"
                )

            print(f"[latex] starting part-{idx + 1}/{len(parts)} conversion...")
            try:
                content = _llama_call_with_fallback([
                    {"role": "system", "content": LATEX_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content},
                ])
            except Exception as e:
                print(f"[latex] error on part-{idx + 1}: {e}")
                raise

            if is_first:
                subject, full_doc = _parse_latex_response(content)
                print(f"[latex] part-1 done, length={len(full_doc)}")
            else:
                merged_body_parts.append(content.strip())
                print(f"[latex] part-{idx + 1} done, length={len(content)}")

            # ↑ increased delay: 20 seconds (extra safety for TPM limit)
            if idx < len(parts) - 1:
                time.sleep(20)

        end_tag = r"\end{document}"
        extra_body = "\n".join(merged_body_parts)
        if end_tag in full_doc:
            merged = full_doc.replace(end_tag, f"\n{extra_body}\n{end_tag}", 1)
        else:
            merged = full_doc + "\n" + extra_body

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
        if not GROQ_API_KEY and not OPENROUTER_API_KEY:
            raise EnvironmentError("GROQ_API_KEY or OPENROUTER_API_KEY must be set.")

        client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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
            if not client:
                raise RuntimeError("Groq client not available for transcription. Need GROQ_API_KEY for Whisper.")
            transcription = transcribe_chunks(chunks, client, job_id=job_id)
            print(f"[pipeline] step 3/4 — done, transcript length={len(transcription)}")
        except Exception as e:
            print(f"[pipeline] step 3/4 — transcription failed: {e}")
            raise

        print("[pipeline] step 4/4 — converting to LaTeX")
        _set_progress("converting to LaTeX")
        try:
            latex_result = to_latex(transcription)
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
    # Background workers (with webhook)
    # ---------------------------------------------------------------------------


    def _run_youtube_job(job_id: str, youtube_url: str, webhook_url: str = "", max_duration: int = 7200) -> None:
        tmp_dir = make_tmp()
        try:
            duration = get_video_duration(youtube_url)
            if duration > max_duration:
                hours = max_duration // 3600
                _fail_job(job_id, f"الفيديو طويل جداً، الحد الأقصى {hours} ساعات حالياً")
                _send_webhook(job_id)
                return
            audio_path = download_youtube_audio(youtube_url, tmp_dir)
            result = process_audio(audio_path, tmp_dir, job_id=job_id)
            _finish_job(job_id, result)
            _send_webhook(job_id)
        except Exception as exc:
            _fail_job(job_id, str(exc))
            _send_webhook(job_id)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


    def _run_upload_job(job_id: str, audio_path: Path, tmp_dir: Path) -> None:
        try:
            result = process_audio(audio_path, tmp_dir, job_id=job_id)
            _finish_job(job_id, result)
            _send_webhook(job_id)
        except Exception as exc:
            _fail_job(job_id, str(exc))
            _send_webhook(job_id)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


    # ---------------------------------------------------------------------------
    # Routes (with optional webhook_url)
    # ---------------------------------------------------------------------------


    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})


    @app.route("/api/transcribe", methods=["POST"])
    def transcribe_youtube():
        """Kick off a background YouTube transcription job; return job_id immediately."""
        data = request.get_json(silent=True) or {}
        youtube_url = data.get("youtube_url", "").strip()
        webhook_url = data.get("webhook_url", "").strip()

        if not youtube_url:
            return jsonify({"error": "youtube_url is required"}), 400

        job_id = _new_job(webhook_url=webhook_url)
        thread = threading.Thread(
            target=_run_youtube_job,
            args=(job_id, youtube_url, webhook_url),
            daemon=True,
        )
        thread.start()

        return jsonify({"job_id": job_id, "status": "processing"}), 202


    @app.route("/api/transcribe-async", methods=["POST"])
    def transcribe_youtube_async():
        """Long-video async transcription — supports up to 6 hours. Returns job_id immediately.

        Accepts either:
        - JSON body:              {"youtube_url": "...", "webhook_url": "..."}
        - multipart/form-data:   file field named 'file' + webhook_url form field
        """
        # ── File upload path ────────────────────────────────────────────────────
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                return jsonify({"error": "No file part in request"}), 400
            uploaded = request.files["file"]
            if not uploaded.filename:
                return jsonify({"error": "Empty filename"}), 400

            webhook_url = request.form.get("webhook_url", "").strip()

            tmp_dir = make_tmp()
            suffix = Path(uploaded.filename).suffix or ".mp3"
            audio_path = tmp_dir / f"upload{suffix}"
            uploaded.save(str(audio_path))

            job_id = _new_job(webhook_url=webhook_url)
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
        webhook_url = data.get("webhook_url", "").strip()

        if not youtube_url:
            return jsonify({"error": "Provide either a youtube_url (JSON) or a file (multipart)"}), 400

        job_id = _new_job(webhook_url=webhook_url)
        thread = threading.Thread(
            target=_run_youtube_job,
            args=(job_id, youtube_url, webhook_url, 21600),  # 6-hour limit
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id, "status": "processing"}), 202


    @app.route("/api/upload", methods=["POST"])
    def upload_audio():
        """Kick off a background upload transcription job; return job_id immediately."""
        webhook_url = ""

        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            webhook_url = data.get("webhook_url", "").strip()

        if request.content_type and "multipart/form-data" in request.content_type:
            webhook_url = request.form.get("webhook_url", "").strip()

        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "Empty filename"}), 400

        tmp_dir = make_tmp()
        suffix = Path(uploaded.filename).suffix or ".mp3"
        audio_path = tmp_dir / f"upload{suffix}"
        uploaded.save(str(audio_path))

        job_id = _new_job(webhook_url=webhook_url)
        thread = threading.Thread(
            target=_run_upload_job,
            args=(job_id, audio_path, tmp_dir),
            daemon=True,
        )
        thread.start()

        return jsonify({"job_id": job_id, "status": "processing"}), 202


    @app.route("/api/status/<job_id>", methods=["GET"])
    def job_status(job_id: str):
        """Poll for job results (still works if you prefer polling over webhook)."""
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