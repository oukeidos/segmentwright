# Segmentwright

Segmentwright is a subtitle pipeline that starts from speech transcripts. It can split transcript
text into subtitle segments with a local model, then uses forced alignment to assign timestamps and
write an SRT file.

Segmentwright addresses a practical gap in Gemini 3-class speech transcription: the text can be
accurate, while direct timestamp output is often unreliable for subtitles. It treats the ASR model
as a transcript generator, then uses local segmentation and CTC forced alignment to build SRT
timestamps.

It can run in three ways:

1. Send audio to Gemini, receive segmented transcript lines, then align them locally.
2. Send audio to Gemini, receive plain text, split it locally, then align it locally.
3. Read an external chunk-level transcript JSON, split it locally, then align it locally.

Current limitation: forced alignment supports Japanese and English only.

## Pipeline Overview

Segmentwright separates transcription from timing.

```text
media
  -> normalize audio
  -> detect speech regions
  -> get transcript chunks
  -> split transcript text into subtitle segments
  -> forced-align segments to audio
  -> write SRT
```

The transcript source can be Gemini or an external JSON file. Gemini is never asked to produce final
timestamps. Timestamp recovery is done locally with CTC forced-alignment models.

Supported languages:

| Language | Code | Default alignment model |
| --- | --- | --- |
| Japanese | `ja` | `nvidia/parakeet-tdt_ctc-0.6b-ja` |
| English | `en` | `nvidia/parakeet-ctc-0.6b` |

Product presets:

| Preset | Transcription | Response | Segmentation |
| --- | --- | --- | --- |
| `quality-baseline` | `gemini-3.5-flash` | JSON `lines` | direct alignment |
| `cheaper-candidate` | `gemini-3-flash-preview` | plain text | local resegmentation |
| `lite-experimental` | `gemini-3.1-flash-lite-preview` | plain text | local resegmentation |

The default preset is `quality-baseline`.

## Installation

Segmentwright uses Python 3.12 and `uv`.

### 1. Clone the repository

```bash
git clone https://github.com/oukeidos/segmentwright.git
cd segmentwright
```

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Install FFmpeg

Install FFmpeg separately and make sure `ffmpeg` and `ffprobe` are available on `PATH`.

### 4. Configure a Gemini API key

Built-in Gemini transcription needs a Gemini API key. External transcript mode does not call Gemini
and does not need this key.

macOS or Linux:

```bash
mkdir -p ~/.segmentwright
cp .env.example ~/.segmentwright/.env
chmod 600 ~/.segmentwright/.env
```

Env file path:

```text
~/.segmentwright/.env
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.segmentwright"
Copy-Item .env.example "$env:USERPROFILE\.segmentwright\.env"
```

Env file path:

```text
%USERPROFILE%\.segmentwright\.env
```

On shared machines, restrict this file to your user account with your operating system's normal file
permission tools.

Set one of these values in the env file:

```text
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
```

### 5. Install Ollama for local resegmentation

Local resegmentation uses Ollama. Install Ollama separately, then pull a model:

```bash
ollama pull gemma4:12b
```

`gemma4:12b` is the default local resegmentation model. For lower-spec machines, use
`gemma3:4b-it-qat`:

```bash
ollama pull gemma3:4b-it-qat
```

Segmentwright can use an already-running local Ollama server. If no local server is reachable, it
starts `ollama serve` for the resegmentation stage. It does not install Ollama. It does not pull
models unless `--ollama-auto-pull` is passed.

### 6. First-run model downloads

The Parakeet alignment models are downloaded by the NeMo / Hugging Face stack on first use.

The FireRedVAD model files are also downloaded automatically by default from Hugging Face
(`FireRedTeam/FireRedVAD`) to `~/.cache/segmentwright/FireRedVAD/VAD`. Use
`--no-firered-vad-auto-download` to disable this, or `--firered-vad-model-dir` to use a prepared
local model directory.

## Basic Usage

### Gemini Backend

Default Japanese run:

```bash
uv run segmentwright input.mkv output.srt
```

English run:

```bash
uv run segmentwright input.mkv output.srt --language en
```

Use the lower-cost Gemini 3 Flash Preview preset:

```bash
uv run segmentwright input.mkv output.srt --preset cheaper-candidate
```

Use the Gemini 3.1 Flash Lite experimental preset:

```bash
uv run segmentwright input.mkv output.srt --preset lite-experimental
```

Preserve diagnostic artifacts:

```bash
uv run segmentwright input.mkv output.srt --artifacts artifacts/run-001
```

Without `--artifacts`, intermediate audio chunks, transcripts, logs, and diagnostics are temporary
and are deleted after the run.

### External Transcript Scripts

You can use any external ASR script if it writes a chunk-level transcript JSON file.

```bash
uv run segmentwright input.mkv output.srt --transcript transcript.json
```

In this mode Segmentwright skips Gemini, reads transcript chunks, runs local resegmentation, runs
forced alignment, and writes SRT.

For built-in Gemini transcription, Segmentwright plans chunks at 180 seconds by default
(`--target-chunk-sec 180`, `--max-chunk-sec 180`). External transcript JSON uses the chunk times you
provide. Very long chunks can increase CTC forced-alignment VRAM use, so keep external transcript
chunks reasonably sized for your GPU. If alignment runs out of memory, split the transcript into
shorter chunks.

Transcript JSON format:

```json
{
  "version": 1,
  "language": "ja",
  "chunks": [
    {
      "id": "chunk_0001",
      "start": 0.0,
      "end": 145.792,
      "text": "ここにこの区間の転写テキスト..."
    }
  ]
}
```

Required fields:

| Field | Meaning |
| --- | --- |
| `version` | Format version. Use `1`. |
| `language` | `ja` or `en`. |
| `chunks` | Ordered transcript chunks. |
| `start` | Chunk start time in media seconds. |
| `end` | Chunk end time in media seconds. |
| `text` | Transcript text for that chunk. Empty text is allowed for no-speech chunks. |

Optional chunk fields such as `source`, `confidence`, or `notes` may be present. Segmentwright keeps
only the fields it needs.

## Flag Reference

### Main Options

| Flag | Description |
| --- | --- |
| `INPUT` | Input audio or video file. Supported extensions: `.mp4`, `.mkv`, `.mov`, `.webm`, `.mp3`, `.m4a`, `.wav`. |
| `OUTPUT` | Output `.srt` path. |
| `--preset` | Select `quality-baseline`, `cheaper-candidate`, or `lite-experimental`. Default: `quality-baseline`. |
| `--language` | Source speech language: `ja` or `en`. Default: `ja`, unless `--transcript` provides a language. |
| `--overwrite` | Replace an existing output file, and allow reuse of a non-empty explicit artifacts directory. |
| `--verbose` | Show detailed Segmentwright and third-party model logs in the terminal. |
| `--device` | Local alignment device: `auto`, `cuda`, or `cpu`. Default: `auto`. |

### Gemini Transcription

These flags apply only when `--transcript` is not used.

| Flag | Description |
| --- | --- |
| `--env-file` | Env file for `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Default: `~/.segmentwright/.env`. |
| `--gemini-model` | Override the preset Gemini model. |
| `--prompt-preset` | Built-in prompt: `subtitle-lines` or `plain-transcript`. |
| `--prompt-file` | Read a custom UTF-8 prompt from a file. Mutually exclusive with `--prompt-text`. |
| `--prompt-text` | Use a custom prompt string. Mutually exclusive with `--prompt-file`. |
| `--response-format` | Gemini response format: `json_lines` or `plain_text`. |
| `--gemini-thinking-level`, `--thinking-level` | Optional Gemini thinking level: `minimal`, `low`, `medium`, or `high`. |
| `--gemini-temperature`, `--temperature` | Optional Gemini sampling temperature. Valid range: `0.0` to `2.0`. |

### External Transcript Input

| Flag | Description |
| --- | --- |
| `--transcript` | Use a pre-generated chunk-level transcript JSON and skip Gemini. |
| `--strict-transcript-chunks` | Treat transcript chunk validation warnings as errors. |
| `--transcript-chunk-warning-sec` | Warn when an imported transcript chunk is longer than this many seconds. Default: same as `--max-chunk-sec`. |

### Segmentation and Ollama

| Flag | Description |
| --- | --- |
| `--segmentation` | Select `direct` or `resegmentation`. `--transcript` defaults to `resegmentation`. |
| `--resegmentation-model`, `--ollama-model` | Ollama model for local resegmentation. Default: `gemma4:12b`. |
| `--resegmentation-ollama-url` | Ollama base URL. Default: `http://localhost:11434`. |
| `--resegmentation-prompt` | Local segmentation prompt: `short`, `medium`, `long`, or `full`. Default: `medium`. |
| `--resegmentation-max-retries` | Maximum local resegmentation retries after a quality-gate failure. Default: `3`. |
| `--ollama-auto-pull` | Pull the configured Ollama model automatically if it is missing. |
| `--ollama-start-timeout-sec` | Seconds to wait for an auto-started Ollama server. Default: `30`. |

For resegmentation, Segmentwright sends `think=false` and `num_ctx=4096` to Ollama. For
`gemma3:4b-it-qat` and `gemma3:12b-it-qat`, it also sends `temperature=0.0`. Other sampling
settings are left to the selected model defaults.

### Forced Alignment

| Flag | Description |
| --- | --- |
| `--alignment-model` | Override the forced-alignment model. By default, Japanese uses `nvidia/parakeet-tdt_ctc-0.6b-ja`; English uses `nvidia/parakeet-ctc-0.6b`. |

### VAD and Chunk Planning

| Flag | Description |
| --- | --- |
| `--target-chunk-sec` | Target chunk duration in seconds. Default: `180`. |
| `--max-chunk-sec` | Maximum chunk duration in seconds. Default: `180`. |
| `--min-retry-split-sec` | Minimum chunk length used when splitting a retry item. Default: `30`. |
| `--firered-vad-model-dir` | Local FireRedVAD model directory. Default: `~/.cache/segmentwright/FireRedVAD/VAD`. |
| `--no-firered-vad-auto-download` | Disable automatic FireRedVAD model download. |

### QA and Retry

| Flag | Description |
| --- | --- |
| `--long-omission-sec` | Long omission threshold in seconds. Default: `60`. |
| `--coverage-gap-merge-sec` | Merge nearby uncovered gaps within this many seconds. Default: `2`. |
| `--max-retries` | Maximum Gemini retry passes for flagged chunks. Default: `1`. |
| `--max-split-depth` | Maximum split depth for retry chunks. Default: `2`. |
| `--max-retry-work-items` | Optional limit on additional retry children. Initial chunks do not count; default: unlimited. |

### Artifacts, Cache, and Logging

| Flag | Description |
| --- | --- |
| `--artifacts` | Preserve diagnostic artifacts and intermediate work files at this path. Default: temporary workdir, deleted after the run. |
| `--cache-dir` | Store transcript cache entries outside the temporary workdir. |
| `--no-cache` | Disable transcript cache reads and writes. |
| `--force-gemini` | Force fresh Gemini calls even when cached transcripts exist. |

Important preserved artifact files include `run_config.json`, `run_summary.json`,
`chunk_manifest.json`, `transcripts.json`, `token_usage.json`, `alignment.json`, `qa_report.json`,
`retry_trace.json`, and `errors.json` when a run fails. Retry configuration is recorded in
`run_config.json`; generation-level scheduling decisions and budget skips are recorded in
`retry_trace.json`; aggregate retry counts are recorded in `run_summary.json`.

## Troubleshooting

### FFmpeg is not found

Install FFmpeg and make sure both `ffmpeg` and `ffprobe` are available on `PATH`.

### Gemini key is not found

Check that `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set in `~/.segmentwright/.env`, or pass a
different file with `--env-file`. `--transcript` mode does not need a Gemini key.

### Ollama model is missing

Pull the model before running a preset that uses local resegmentation:

```bash
ollama pull gemma4:12b
```

Or let Segmentwright pull it:

```bash
uv run segmentwright input.mkv output.srt --preset cheaper-candidate --ollama-auto-pull
```

### Local resegmentation runs out of RAM or VRAM

Use a smaller Ollama model:

```bash
uv run segmentwright input.mkv output.srt \
  --preset cheaper-candidate \
  --ollama-model gemma3:4b-it-qat
```

You can also use `--segmentation direct` to skip local resegmentation when your transcript already
has acceptable line breaks.

### Forced alignment runs out of VRAM

Reduce chunk size so the alignment model processes shorter audio windows:

```bash
uv run segmentwright input.mkv output.srt \
  --target-chunk-sec 60 \
  --max-chunk-sec 60
```

If GPU memory is still too small, try CPU alignment:

```bash
uv run segmentwright input.mkv output.srt --device cpu
```

CPU alignment is usually slower.

### First run is slow

The first run may download FireRedVAD and Parakeet model files. Later runs reuse local caches unless
you change cache locations or remove the downloaded files.

### Output file or artifacts directory already exists

Pass `--overwrite` if you want Segmentwright to replace an existing output file or reuse a non-empty
explicit artifacts directory.
