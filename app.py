from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "Clip Finder — Best YouTube Timestamps"
DEFAULT_MODEL = os.getenv("CLIP_FINDER_MODEL", "gpt-5-mini")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")


@dataclass
class TranscriptLine:
    start: float
    end: float
    text: str


def parse_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Valid YouTube URL nahi mili. Example: https://www.youtube.com/watch?v=VIDEO_ID")


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return float(raw)
    parts = raw.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value}") from exc
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    raise ValueError(f"Invalid timestamp: {value}")


def get_video_metadata(video_id: str, url: str) -> tuple[str, float | None]:
    title = f"YouTube video {video_id}"
    duration: float | None = None
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=15,
        )
        response.raise_for_status()
        title = response.json().get("title") or title
    except Exception:
        pass

    try:
        import yt_dlp

        options = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title") or title
            duration = float(info["duration"]) if info.get("duration") else None
    except Exception:
        pass
    return title, duration


def _parse_vtt(path: str) -> list[TranscriptLine]:
    timestamp_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s+-->\s+"
        r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )

    def vtt_seconds(value: str) -> float:
        parts = value.replace(",", ".").split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

    lines: list[TranscriptLine] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        content = handle.read().replace("\r", "")
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        match = timestamp_re.search(block)
        if not match:
            continue
        text_lines = block[match.end():].splitlines()
        text = " ".join(line.strip() for line in text_lines if line.strip())
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            start = vtt_seconds(match.group("start"))
            end = vtt_seconds(match.group("end"))
            lines.append(TranscriptLine(start=start, end=max(end, start + 0.1), text=text))
    return lines


def get_transcript(video_id: str) -> list[TranscriptLine]:
    from youtube_transcript_api import YouTubeTranscriptApi

    raw_items: list[Any] = []
    errors: list[str] = []
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=["en", "ur", "hi"])
            raw_items = list(fetched)
        elif hasattr(api, "get_transcript"):
            raw_items = api.get_transcript(video_id, languages=["en", "ur", "hi"])
    except Exception as first_error:
        errors.append(str(first_error))

    lines: list[TranscriptLine] = []
    for item in raw_items:
        if hasattr(item, "text"):
            text = str(item.text)
            start = float(item.start)
            duration = float(item.duration)
        else:
            text = str(item.get("text", ""))
            start = float(item.get("start", 0))
            duration = float(item.get("duration", 0))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(TranscriptLine(start=start, end=start + max(duration, 0.1), text=text))
    if lines:
        return lines

    # Fallback: yt-dlp can retrieve public/manual/automatic VTT captions when the
    # transcript endpoint is blocked by a cloud IP or rate limit.
    try:
        import glob
        import tempfile
        import yt_dlp

        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, "captions.%(ext)s")
            options = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB", "ur", "hi"],
                "subtitlesformat": "vtt",
                "outtmpl": output_template,
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            subtitle_files = glob.glob(os.path.join(temp_dir, "*.vtt"))
            for subtitle_file in subtitle_files:
                lines = _parse_vtt(subtitle_file)
                if lines:
                    return lines
    except Exception as fallback_error:
        errors.append(str(fallback_error))

    raise RuntimeError(
        "Is video ka transcript/captions retrieve nahi ho saka. Video par captions enabled honi chahiye. "
        + " | ".join(errors[-2:])
    )


def compact_transcript(lines: list[TranscriptLine]) -> str:
    return "\n".join(
        f"[{format_time(line.start)}-{format_time(line.end)}] {line.text}" for line in lines
    )


def build_prompt(
    transcript: str,
    video_title: str,
    requested_count: int,
    min_duration: int,
    max_duration: int,
) -> str:
    return f"""You are an expert short-form video editor. Analyze this YouTube transcript and find the strongest self-contained moments for TikTok, Instagram Reels, and YouTube Shorts.

Video title: {video_title}
Return exactly {requested_count} ranked candidates if the transcript contains enough strong moments. Every clip MUST be at least {min_duration} seconds and no longer than {max_duration} seconds. Never return a 2–5 second fragment, a single sentence without context, or an isolated phrase. If there are not enough strong moments, return fewer clips rather than short fragments.

A good clip has: a hook in the first 1-3 seconds, a clear idea or conflict, a satisfying payoff/reaction, and enough context to stand alone. Include the setup before the hook and the reaction, conclusion, or payoff after it. Prioritize surprising claims, controversy, emotional stories, strong opinions, jokes, reveals, arguments, practical advice, and moments that will generate comments. Do not invent words or topics that are absent from the transcript. Timestamps must fall inside the transcript.

Output JSON only in this shape:
{{
  "clips": [
    {{
      "rank": 1,
      "start": "MM:SS",
      "end": "MM:SS",
      "title": "short clip title",
      "hook": "exact or near-exact opening wording from the transcript",
      "core_idea": "what the clip is about",
      "payoff": "what happens at the end",
      "platform": "TikTok / Instagram Reels / YouTube Shorts",
      "virality_score": 1,
      "clarity_score": 1,
      "retention_score": 1,
      "editor_note": "how to cut or caption it"
    }}
  ]
}}

Transcript:
{transcript}
"""


def analyze_transcript(
    transcript: str,
    title: str,
    count: int,
    min_duration: int,
    max_duration: int,
    model: str,
    provider: str,
) -> dict[str, Any]:
    if provider == "NVIDIA NIM":
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = NVIDIA_BASE_URL
        key_name = "NVIDIA_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        key_name = "OPENAI_API_KEY"
    if not api_key:
        raise RuntimeError(f"{key_name} set nahi hai. .env file mein apni API key add karein.")

    client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You select precise, factual short-form video moments. Output valid JSON only.",
            },
            {"role": "user", "content": build_prompt(transcript, title, count, min_duration, max_duration)},
        ],
        max_completion_tokens=6000,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def normalize_clips(
    data: dict[str, Any],
    transcript_duration: float,
    video_duration: float | None,
    min_duration: int,
    max_duration: int,
) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    upper_bound = video_duration or transcript_duration
    for index, raw in enumerate(data.get("clips", []), start=1):
        try:
            start = max(0.0, parse_time(raw.get("start", 0)))
            end = min(upper_bound, parse_time(raw.get("end", start)))
        except (TypeError, ValueError):
            continue
        clip_length = end - start
        if end <= start or clip_length < min_duration or clip_length > max_duration + 2:
            continue
        item = dict(raw)
        item["rank"] = len(clips) + 1
        item["start_seconds"] = round(start, 2)
        item["end_seconds"] = round(end, 2)
        item["timestamp"] = f"{format_time(start)}–{format_time(end)}"
        for field in ("virality_score", "clarity_score", "retention_score"):
            try:
                item[field] = max(1, min(10, int(item.get(field, 7))))
            except (TypeError, ValueError):
                item[field] = 7
        clips.append(item)
    return clips


def report_markdown(title: str, url: str, clips: list[dict[str, Any]]) -> str:
    rows = [f"# Clip Finder Report\n\n**Video:** [{title}]({url})\n"]
    for clip in clips:
        rows.append(
            f"## {clip['rank']}. {clip.get('title', 'Untitled clip')} — `{clip['timestamp']}`\n\n"
            f"**Hook:** {clip.get('hook', 'N/A')}\n\n"
            f"**Core idea:** {clip.get('core_idea', 'N/A')}\n\n"
            f"**Payoff:** {clip.get('payoff', 'N/A')}\n\n"
            f"**Platform:** {clip.get('platform', 'TikTok / Reels / Shorts')}  \n"
            f"**Scores:** Virality {clip.get('virality_score', 7)}/10 · "
            f"Clarity {clip.get('clarity_score', 7)}/10 · "
            f"Retention {clip.get('retention_score', 7)}/10\n\n"
            f"**Editor note:** {clip.get('editor_note', 'Trim pauses and begin on the hook.')}\n"
        )
    return "\n".join(rows)


st.set_page_config(page_title=APP_TITLE, page_icon="✂️", layout="wide")
st.title("Clip Finder")
st.caption("YouTube link paste karein — best short-form timestamps, hooks aur scores paayein.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("AI provider", ["OpenAI-compatible", "NVIDIA NIM"])
    if provider == "NVIDIA NIM":
        model = st.text_input("NVIDIA model", value=NVIDIA_DEFAULT_MODEL)
        st.caption("NVIDIA_API_KEY .env file mein add karein.")
    else:
        model = st.text_input("AI model", value=DEFAULT_MODEL)
        st.caption("OPENAI_API_KEY .env file mein add karein.")
    count = st.slider("Kitne clips chahiye?", min_value=3, max_value=15, value=8)
    min_duration = st.slider("Minimum clip duration (seconds)", min_value=15, max_value=60, value=30, step=5)
    max_duration = st.slider("Maximum clip duration (seconds)", min_value=30, max_value=180, value=90, step=5)
    st.info("Tool complete hook + discussion + payoff wale clips select karta hai. Final export se pehle 5–10 seconds ka context editor mein verify karein.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
run = st.button("Find Best Timestamps", type="primary", use_container_width=True)

if run:
    if not url.strip():
        st.error("Pehle YouTube link paste karein.")
        st.stop()
    try:
        video_id = parse_video_id(url.strip())
        with st.status("Video ko analyze kiya ja raha hai...", expanded=True) as status:
            st.write("Video metadata read ho raha hai...")
            title, video_duration = get_video_metadata(video_id, url.strip())
            st.write("Transcript nikala ja raha hai...")
            lines = get_transcript(video_id)
            transcript = compact_transcript(lines)
            transcript_duration = max(line.end for line in lines)
            st.write("AI strong moments select kar raha hai...")
            raw_result = analyze_transcript(transcript, title, count, min_duration, max_duration, model, provider)
            clips = normalize_clips(raw_result, transcript_duration, video_duration, min_duration, max_duration)
            status.update(label="Analysis complete", state="complete")

        if not clips:
            st.warning("Strong timestamp candidates nahi milay. Captions ya transcript quality check karein.")
            st.stop()

        st.subheader(title)
        duration_text = format_time(video_duration or transcript_duration)
        st.caption(f"Approximate duration: {duration_text} · {len(clips)} clips found")

        display_rows = [
            {
                "Rank": clip["rank"],
                "Timestamp": clip["timestamp"],
                "Title": clip.get("title", ""),
                "Platform": clip.get("platform", ""),
                "Virality": clip.get("virality_score", 7),
                "Clarity": clip.get("clarity_score", 7),
                "Retention": clip.get("retention_score", 7),
            }
            for clip in clips
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        for clip in clips:
            with st.expander(f"{clip['rank']}. {clip.get('title', 'Untitled')} — {clip['timestamp']}"):
                st.markdown(f"**Hook:** {clip.get('hook', 'N/A')}")
                st.markdown(f"**Core idea:** {clip.get('core_idea', 'N/A')}")
                st.markdown(f"**Payoff:** {clip.get('payoff', 'N/A')}")
                st.markdown(f"**Editor note:** {clip.get('editor_note', 'N/A')}")
                st.write(
                    f"Platform: {clip.get('platform', 'TikTok / Reels / Shorts')} · "
                    f"Virality: {clip.get('virality_score', 7)}/10 · "
                    f"Clarity: {clip.get('clarity_score', 7)}/10 · "
                    f"Retention: {clip.get('retention_score', 7)}/10"
                )

        report = report_markdown(title, url.strip(), clips)
        st.download_button(
            "Download report (Markdown)",
            data=report,
            file_name="clip_finder_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.warning("AI timestamps editorial estimates hain. Upload se pehle exact start/end frame aur copyright permissions verify karein.")
    except Exception as exc:
        st.error(str(exc))
        st.info("Agar video mein captions nahi hain, pehle captions/transcript available video try karein.")

st.divider()
st.caption("Made for clip research. This tool does not bypass copyright restrictions or platform rules.")
