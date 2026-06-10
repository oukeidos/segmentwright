# DISCLAIMER

`segmentwright` generates aligned subtitle files from speech media. It can use
built-in Gemini transcription or a user-provided chunk-level transcript, then
uses local processing for segmentation, forced alignment, subtitle assembly, and
diagnostics.

By using this project, you accept the following.

This document is a practical notice only. It does not replace `LICENSE`,
`THIRD_PARTY_NOTICES.md`, upstream package licenses, model licenses, API terms,
or the terms of any service used with this project.

## 1) Intended Use

`segmentwright` is designed for assisted subtitle generation and timing recovery.

- Input: audio or video media containing speech
- Optional input: chunk-level transcript JSON supplied by the user
- Processing: media normalization, VAD/chunk planning, transcription,
  resegmentation, forced alignment, quality checks, and SRT assembly
- Output: text subtitle files such as `SRT`

The output is a working subtitle draft, not a guaranteed final subtitle master.
Manual review is expected.

## 2) No Warranty

The software is provided "as is".

There is no warranty of:

- correctness
- completeness
- fitness for a particular purpose
- availability
- performance
- non-infringement
- compatibility with any particular model, service, driver, operating system, or
  hardware environment

You are responsible for validating both the software and its output before any
real-world use.

## 3) Limitation of Liability

To the maximum extent permitted by law, the authors and contributors are not
liable for any direct, indirect, incidental, special, consequential, or other
damages arising from use of the software, generated subtitles, intermediate
artifacts, model outputs, API calls, local runtimes, or related files.

## 4) Transcription and Subtitle Accuracy

ASR, segmentation, forced alignment, and local LLM resegmentation can be wrong,
incomplete, unstable, or sensitive to model version and runtime configuration.

Examples:

- speech may be missed
- non-speech audio may be hallucinated as speech
- words, names, numbers, or terminology may be misrecognized
- repeated or overlapping speech may be merged incorrectly
- line breaks and subtitle boundaries may be unnatural
- timestamps may be early, late, missing, or attached to the wrong text
- music, noise, radio effects, background speech, or stylized audio may degrade
  output quality
- model reruns may produce different results even with similar settings

Do not use unreviewed output in contexts where subtitle errors can cause legal,
financial, safety, accessibility, compliance, reputational, or other material
harm.

## 5) Models, Services, Dependencies, and Third-Party Terms

`segmentwright` may depend on third-party software, model artifacts, local
runtimes, and remote services. Depending on the selected options, this can
include:

- Google Gemini API for built-in transcription
- Ollama for local LLM resegmentation
- Gemma-family local models selected through Ollama
- NVIDIA NeMo / Parakeet models for forced alignment
- FireRedVAD for local speech-region detection
- FFmpeg / ffprobe host binaries for media handling
- PyTorch, torchaudio, NeMo, Hugging Face Hub, and related Python packages

Third-party licenses, model terms, API terms, usage policies, rate limits, and
billing rules apply to those components. The `segmentwright` license does not
replace them.

See:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- upstream package, model, runtime, and service terms

## 6) API Use, Billing, and Availability

Built-in Gemini transcription sends media chunks and prompt text to the
configured Gemini API endpoint. API behavior, pricing, quotas, rate limits,
retention policies, safety filters, response formats, model availability, and
model names are controlled by the service provider and can change.

You are responsible for:

- configuring API credentials safely
- monitoring token use and API cost
- confirming that your media is allowed to be sent to the selected service
- reviewing the service provider's data handling and retention terms
- handling quota failures, transient errors, model changes, and service outages

Use `--transcript` if you want to skip built-in Gemini transcription and provide
transcript chunks through a local JSON file instead.

## 7) Network and Local Data Handling

Different execution paths use different data locations.

Network access may occur for:

- Gemini transcription API calls
- first-time model downloads through Hugging Face / NeMo stacks
- optional Ollama model pulling when explicitly enabled
- package installation or dependency resolution outside normal runtime

Local execution may create or use:

- input media files
- generated subtitle files
- normalized audio
- media chunks
- transcript cache entries
- token-usage records
- run summaries and diagnostic metadata
- forced-alignment intermediate files
- local model caches
- Ollama model files and server state

By default, `segmentwright` uses a temporary work directory for diagnostic
artifacts and removes it when the run finishes. If you pass `--artifacts PATH`,
those artifacts are preserved at the chosen path for debugging and auditability.

If the media, transcript, subtitles, or logs are sensitive, private, regulated,
or not yours to redistribute, choose storage paths, cache policy, file
permissions, and cleanup procedures accordingly.

## 8) Rights and Legal Compliance

You must have the legal right to process the source media, send it to any
selected service, store intermediate artifacts, and use, modify, or distribute
any resulting subtitle output.

You are responsible for compliance with applicable law and policy, including:

- copyright and neighboring rights
- contract, license, and platform terms
- privacy and data protection rules
- workplace, institutional, customer, or production confidentiality requirements
- accessibility or localization requirements that apply to your workflow

Using this tool does not grant rights to the source media, existing subtitles,
third-party transcripts, model outputs, or any derivative work.

## 9) Operational Limits and Failure Modes

The project may fail because of:

- missing API credentials
- API quota, billing, safety, or availability failures
- missing FFmpeg / ffprobe binaries
- missing Ollama installation or model files
- incompatible package versions
- unsupported operating system behavior
- insufficient CPU, GPU, RAM, or VRAM
- CUDA, driver, PyTorch, torchaudio, or NeMo incompatibilities
- malformed media or transcript JSON
- unsupported language or forced-alignment model mismatch
- model load failures or runtime errors in third-party libraries

Even when execution succeeds, the output can still be wrong. Test on a small
sample before large or production-style runs.

## 10) Security Boundaries

This project does not provide a security sandbox. Treat media files,
transcripts, generated subtitles, model files, caches, and artifacts as
sensitive data when appropriate.

You assume risk for:

- compromised upstream packages, model artifacts, or binaries
- poisoned or stale local caches
- unsafe API key handling
- insecure file permissions
- untrusted media files
- unsafe local environment configuration
- unexpected behavior from local or remote models

Dependency updates, model changes, service-provider changes, and execution
environment changes should be treated as security-relevant events.

## 11) No Professional, Legal, or Accessibility Certification

`segmentwright` is an engineering tool, not a legal, compliance, accessibility,
medical, safety, or editorial certification system.

Generated subtitles may be useful for review, drafting, research, or internal
workflow automation, but you remain responsible for any required human review,
quality assurance, accessibility checks, localization standards, or legal
clearance.

## 12) No Affiliation

This project is not affiliated with, endorsed by, or sponsored by Google,
Gemini, Ollama, Gemma, NVIDIA, NeMo, Hugging Face, FFmpeg, FireRedTeam, PyTorch,
or any other third-party provider referenced by the project.

Names of third-party products and organizations are used only to identify the
relevant software, model artifacts, runtimes, or services.

## 13) User Responsibility

You are responsible for:

- setup and environment configuration
- choosing models, services, and presets
- reviewing subtitle text and timing
- validating output quality for your use case
- monitoring API cost and local resource use
- protecting sensitive media, transcripts, and artifacts
- checking legal, license, service-term, and policy compliance
- deciding whether the tool is suitable for your workflow

If you use this project, you accept these limits and responsibilities.
