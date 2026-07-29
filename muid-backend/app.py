import os
import uuid
import shutil
import subprocess
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
MAX_CHUNK_MS = 8 * 60 * 1000          # 8 minutes in milliseconds
MAX_LATEX_CHARS = 8000                  # max chars sent to LLaMA per batch

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
# Helpers
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


def transcribe_chunks(chunks: list[Path], client: Groq) -> str:
    """Transcribe each chunk and join results."""
    parts = []
    for chunk in chunks:
        parts.append(transcribe_file(chunk, client))
    return "\n".join(parts)


def to_latex(transcription: str, client: Groq) -> dict:
    """Convert transcription text to LaTeX via Groq LLaMA."""
    # Process in batches if text is very long
    text = transcription[:MAX_LATEX_CHARS] if len(transcription) > MAX_LATEX_CHARS else transcription

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": LATEX_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcription:\n\n{text}"},
        ],
        temperature=0.3,
        max_tokens=8192,
    )

    content = response.choices[0].message.content or ""

    subject = ""
    latex = ""
    if "SUBJECT:" in content and "LATEX:" in content:
        subject_part, latex_part = content.split("LATEX:", 1)
        subject = subject_part.replace("SUBJECT:", "").strip()
        latex = latex_part.strip()
    else:
        latex = content

    return {"subject": subject, "latex": latex}


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
        "extractor_args": {"youtube": {"player_client": ["ios", "web"]}},
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded MP3 (extension may vary before postprocessing)
    mp3_files = list(tmp_dir.glob("audio.mp3"))
    if not mp3_files:
        # fallback: any audio file in tmp_dir
        candidates = list(tmp_dir.iterdir())
        if not candidates:
            raise FileNotFoundError("yt-dlp did not produce an output file.")
        return candidates[0]
    return mp3_files[0]


def process_audio(audio_path: Path, tmp_dir: Path) -> dict:
    """Core pipeline: compress → chunk → transcribe → latex."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    client = Groq(api_key=GROQ_API_KEY)

    # Compress
    compressed = tmp_dir / "compressed.mp3"
    compress_audio(audio_path, compressed)

    # Chunk
    chunks = chunk_audio(compressed, tmp_dir)

    # Transcribe
    transcription = transcribe_chunks(chunks, client)

    # LaTeX
    latex_result = to_latex(transcription, client)

    return {
        "transcription": transcription,
        "subject": latex_result["subject"],
        "latex": latex_result["latex"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/transcribe", methods=["POST"])
def transcribe_youtube():
    """Accept a YouTube URL, download, transcribe, and return LaTeX."""
    data = request.get_json(silent=True) or {}
    youtube_url = data.get("youtube_url", "").strip()

    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    tmp_dir = make_tmp()
    try:
        audio_path = download_youtube_audio(youtube_url, tmp_dir)
        result = process_audio(audio_path, tmp_dir)
        return jsonify(result)
    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/upload", methods=["POST"])
def upload_audio():
    """Accept a multipart audio file upload, transcribe, and return LaTeX."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    tmp_dir = make_tmp()
    try:
        suffix = Path(uploaded.filename).suffix or ".mp3"
        audio_path = tmp_dir / f"upload{suffix}"
        uploaded.save(str(audio_path))

        result = process_audio(audio_path, tmp_dir)
        return jsonify(result)
    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
