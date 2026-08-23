# Clip Finder

A local Streamlit tool for finding the strongest short-form timestamps in YouTube videos. Paste a YouTube URL, and the app extracts available captions, sends the timestamped transcript to an AI model, and returns ranked clip candidates.

## Output

For each candidate, the app returns:

- Start and end timestamp
- Clip title
- Opening hook
- Core idea
- Payoff or emotional beat
- Recommended platform
- Virality, clarity, and retention scores
- Editing note

The app also lets you download a Markdown report.

## Requirements

- Python 3.10 or newer
- An OpenAI-compatible API key
- Internet access for YouTube captions and the AI request

## Setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
streamlit run app.py
```

Put your API key in `.env`:

```env
OPENAI_API_KEY=your_key_here
CLIP_FINDER_MODEL=gpt-5-mini
```

For the official OpenAI API, leave `OPENAI_API_BASE` blank. For another OpenAI-compatible provider, set its base URL as documented by that provider.

## NVIDIA NIM / Meta Llama 3.1 70B

The sidebar includes an **NVIDIA NIM** provider option. NVIDIA documents an OpenAI-compatible chat-completions endpoint at `https://integrate.api.nvidia.com/v1`, and the Llama 3.1 70B Instruct model is available with the model ID `meta/llama-3.1-70b-instruct`.

Add this to `.env`:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
```

Get the key from [NVIDIA Build](https://build.nvidia.com/). Start the app, choose **NVIDIA NIM** in the left sidebar, keep the default model, paste a YouTube link, and click **Find Best Timestamps**. The tool uses the same transcript and timestamp workflow; only the AI provider changes.

## How to use

Open the local URL shown by Streamlit, usually `http://localhost:8501`, paste a YouTube link, choose the number of clips and maximum duration, then click **Find Best Timestamps**.

The tool uses the video transcript for semantic analysis. It does not bypass captions, copyright restrictions, or platform rules. AI timestamps should be checked manually in an editor with approximately 5–10 seconds of context before publishing.

## GitHub starter projects reviewed

This project was designed after reviewing these open-source repositories:

- https://github.com/joshua4816470/ai-video-clipper
- https://github.com/obzennn/ReelMind
- https://github.com/ofds/AI-Video-Highlighter
