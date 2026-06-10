# Third-Party Notices

This notice accompanies the public `segmentwright` source release. It identifies the
third-party software, build-time tools, downloaded model artifacts, host runtimes, and
remote services that may be used with the current `segmentwright` product surface.

The exact third-party set varies by operating system, Python version, architecture, CUDA
availability, selected preset, selected language, and whether local resegmentation is used.

## Notice Basis

This notice reflects the current repository state based on:

- `pyproject.toml`
- `uv.lock`
- the `segmentwright` source tree at version `0.1.0`

## First-Party Software

- `segmentwright` is licensed under the MIT License. See `LICENSE`.

## Direct Python Dependencies Declared By This Project

### Core Runtime

| Package | Locked Version | License | Purpose |
| --- | --- | --- | --- |
| `fireredvad` | `0.0.2` | Apache-2.0 | Provides local VAD and audio-event detection used for speech-region discovery. |
| `google-genai` | `2.7.0` | Apache-2.0 | Calls Gemini for built-in ASR transcription when `--transcript` is not used. |
| `huggingface-hub` | `0.36.2` | Apache | Downloads FireRedVAD and NeMo/Hugging Face model artifacts. |
| `nemo-toolkit[asr]` | `2.7.3` | Apache-2.0 | Loads NVIDIA NeMo ASR/CTC models for forced alignment. |
| `pydantic` | `2.13.4` | MIT | Validates external transcript JSON and structured data. |
| `python-dotenv` | `1.2.2` | BSD-3-Clause | Loads local Gemini API key configuration from env files. |
| `torch` | `2.12.0` | BSD-3-Clause | Provides the tensor runtime used by VAD and forced alignment stacks. |
| `torchaudio` | `2.11.0` | BSD-style / PyTorch project terms | Loads and resamples audio for local alignment. |
| `torchcodec` | `0.14.0` | BSD-style / PyTorch project terms | Provides media decoding support in the PyTorch ecosystem. |

### Development Tools

| Package | Locked Version | License | Purpose |
| --- | --- | --- | --- |
| `pytest` | `9.0.3` | MIT | Test runner. |
| `ruff` | `0.15.15` | MIT | Linter and formatter. |

## Locked Supporting Packages

The following packages are currently pinned in `uv.lock` as supporting dependencies of the declared
product surface. They are not all necessarily imported directly by `segmentwright`, and platform or
CUDA resolution may vary.

### Base Runtime, API, Metadata, and Network Support

- `aiohappyeyeballs 2.6.2`
- `aiohttp 3.14.0`
- `aiosignal 1.4.0`
- `annotated-types 0.7.0`
- `anyio 4.13.0`
- `attrs 26.1.0`
- `certifi 2026.5.20`
- `cffi 2.0.0`
- `charset-normalizer 3.4.7`
- `cryptography 48.0.0`
- `distro 1.9.0`
- `filelock 3.29.0`
- `frozenlist 1.8.0`
- `fsspec 2024.12.0`
- `google-auth 2.53.0`
- `h11 0.16.0`
- `hf-xet 1.5.0`
- `httpcore 1.0.9`
- `httpx 0.28.1`
- `idna 3.18`
- `multidict 6.7.1`
- `packaging 24.2`
- `propcache 0.5.2`
- `pyasn1 0.6.3`
- `pyasn1-modules 0.4.2`
- `pycparser 3.0`
- `pydantic-core 2.46.4`
- `PyYAML 6.0.3`
- `requests 2.34.2`
- `sniffio 1.3.1`
- `tenacity 9.1.4`
- `tqdm 4.67.3`
- `typing-extensions 4.15.0`
- `typing-inspection 0.4.2`
- `urllib3 2.7.0`
- `websockets 16.0`
- `yarl 1.24.2`

### Audio, VAD, ASR, Alignment, and ML Support

- `absl-py 2.4.0`
- `accelerate 1.13.0`
- `antlr4-python3-runtime 4.9.3`
- `audioread 3.1.0`
- `braceexpand 0.1.7`
- `cloudpickle 3.1.2`
- `datasets 4.8.5`
- `decorator 5.3.1`
- `dill 0.4.1`
- `editdistance 0.8.1`
- `einops 0.8.2`
- `fiddle 0.3.0`
- `graphviz 0.21`
- `hydra-core 1.3.2`
- `inflect 7.5.0`
- `jiwer 3.1.0`
- `joblib 1.5.3`
- `kaldi-native-fbank 1.22.3`
- `kaldi-python-io 1.2.2`
- `kaldialign 0.9.1`
- `kaldiio 2.18.1`
- `lazy-loader 0.5`
- `lhotse 1.33.0`
- `libcst 1.8.6`
- `librosa 0.11.0`
- `lightning 2.4.0`
- `lightning-utilities 0.15.3`
- `llvmlite 0.47.0`
- `lxml 6.1.1`
- `marshmallow 4.3.0`
- `mediapy 1.1.6`
- `ml-dtypes 0.5.4`
- `msgpack 1.1.2`
- `multiprocess 0.70.19`
- `narwhals 2.22.0`
- `networkx 3.6.1`
- `numba 0.65.1`
- `numexpr 2.13.1`
- `numpy 2.4.6`
- `omegaconf 2.3.0`
- `onnx 1.21.0`
- `optuna 4.9.0`
- `pandas 3.0.3`
- `peft 0.19.1`
- `pillow 12.2.0`
- `platformdirs 4.10.0`
- `pooch 1.9.0`
- `protobuf 5.29.6`
- `psutil 7.2.2`
- `pyannote-core 6.0.1`
- `pyannote-database 6.1.1`
- `pyannote-metrics 4.1`
- `pyarrow 24.0.0`
- `pydub 0.25.1`
- `pyloudnorm 0.2.0`
- `pytorch-lightning 2.6.5`
- `rapidfuzz 3.14.5`
- `regex 2026.5.9`
- `resampy 0.4.3`
- `ruamel-yaml 0.19.1`
- `sacrebleu 2.6.0`
- `sacremoses 0.1.1`
- `safetensors 0.7.0`
- `scikit-learn 1.9.0`
- `scipy 1.17.1`
- `sentencepiece 0.2.1`
- `soundfile 0.13.1`
- `sox 1.5.0`
- `soxr 1.1.0`
- `sympy 1.14.0`
- `tensorboard 2.20.0`
- `tensorboard-data-server 0.7.2`
- `text-unidecode 1.3`
- `textgrid 1.6.1`
- `threadpoolctl 3.6.0`
- `tokenizers 0.22.2`
- `torchmetrics 1.9.0`
- `transformers 4.57.6`
- `triton 3.7.0`
- `wandb 0.27.0`
- `webdataset 1.0.2`
- `whisper-normalizer 0.1.12`
- `wrapt 2.2.1`
- `xxhash 3.7.0`

### Optional CUDA and NVIDIA Runtime Packages

These packages may be present when CUDA-enabled PyTorch, NeMo, or CUDA toolkit wheels are installed:

- `cuda-bindings 13.3.1`
- `cuda-pathfinder 1.5.5`
- `cuda-toolkit 13.0.2`
- `nvidia-cublas 13.1.1.3`
- `nvidia-cuda-cupti 13.0.85`
- `nvidia-cuda-nvrtc 13.0.88`
- `nvidia-cuda-runtime 13.0.96`
- `nvidia-cudnn-cu13 9.20.0.48`
- `nvidia-cufft 12.0.0.61`
- `nvidia-cufile 1.15.1.6`
- `nvidia-curand 10.4.0.35`
- `nvidia-cusolver 12.0.4.66`
- `nvidia-cusparse 12.6.3.3`
- `nvidia-cusparselt-cu13 0.8.1`
- `nvidia-nccl-cu13 2.29.7`
- `nvidia-nvjitlink 13.0.88`
- `nvidia-nvshmem-cu13 3.4.5`
- `nvidia-nvtx 13.0.85`

### Source Release Build, Test, and Lint Tools

These tools are used for source packaging, verification, and local development. They are not part of
the normal end-user runtime as standalone application dependencies:

| Package | Locked Version | License |
| --- | --- | --- |
| `iniconfig` | `2.3.0` | MIT |
| `pluggy` | `1.6.0` | MIT |
| `Pygments` | `2.20.0` | BSD-2-Clause |
| `pytest` | `9.0.3` | MIT |
| `ruff` | `0.15.15` | MIT |
| `setuptools` | `81.0.0` | MIT |

## Host-Provided Build Tools, Runtimes, and Remote Services

| Component | Source | License / Terms | Notes |
| --- | --- | --- | --- |
| `ffmpeg` / `ffprobe` | User-provided host binaries | The license terms for the exact FFmpeg build installed on the host apply. FFmpeg builds are commonly LGPL-2.1-or-later or GPL, depending on build configuration. | Required for media normalization, probing, and chunk extraction. Segmentwright does not vendor FFmpeg binaries. |
| Ollama server and CLI | User-installed Ollama runtime | Ollama project license and terms apply to the installed runtime. | Used only when local resegmentation is active. Segmentwright can start a local `ollama serve` process, but it does not install Ollama. |
| Google Generative Language API (`Gemini`) | Remote Google service | Google API, model, and service terms apply. | Used for built-in Gemini transcription. `--transcript` mode skips this service. No Gemini SDK model weights are redistributed by this repository. |
| Hugging Face Hub | Remote Hugging Face service | Hugging Face service and repository terms apply. | Used to download FireRedVAD model files and NeMo/Hugging Face model artifacts. |

## Downloaded Runtime, Model, and Artifact Sources

`segmentwright` does not vendor the following runtime files, model files, or derived artifacts in the
repository. They are downloaded or materialized locally when the relevant code path is used.

| Artifact | Source | License / Terms | Notes |
| --- | --- | --- | --- |
| FireRedVAD model files (`model.pth.tar`, `cmvn.ark`) | <https://huggingface.co/FireRedTeam/FireRedVAD> | Apache-2.0 according to the upstream model card | Downloaded automatically by default if the configured FireRedVAD model directory is missing. |
| Japanese forced-alignment model `nvidia/parakeet-tdt_ctc-0.6b-ja` | <https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja> | CC-BY-4.0 according to the upstream model card | Default Japanese CTC-capable NeMo model used for forced alignment. |
| English forced-alignment model `nvidia/parakeet-ctc-0.6b` | <https://huggingface.co/nvidia/parakeet-ctc-0.6b> | CC-BY-4.0 according to the upstream model card | Default English CTC model used for forced alignment. |
| Default local resegmentation model `gemma4:12b` | <https://ollama.com/library/gemma4:12b> | Google Gemma terms apply to the model weights; Ollama terms apply to Ollama distribution and runtime | Used by default when `--segmentation resegmentation` is active. Segmentwright does not vendor or pull it unless the user enables auto-pull. |
| Lower-spec resegmentation model `gemma3:4b-it-qat` | <https://ollama.com/library/gemma3:4b-it-qat> | Google Gemma terms apply to the model weights; Ollama terms apply to Ollama distribution and runtime | Optional lower-memory local resegmentation model. |
| Conservative comparison model `gemma3:12b-it-qat` | <https://ollama.com/library/gemma3:12b-it-qat> | Google Gemma terms apply to the model weights; Ollama terms apply to Ollama distribution and runtime | Supported for older comparison runs and manual overrides. |
| Custom Gemini, Ollama, or alignment model IDs | User-selected model provider or local model source | Terms for the selected upstream model and service apply | Segmentwright exposes model override flags. The user is responsible for reviewing the terms for custom models. |

## Redistribution Guidance

- If you distribute `segmentwright` source, preserve `LICENSE` and this notice.
- If you distribute an environment, container, installer, or frozen application that bundles Python
  wheels, preserve the license files and notices included in those wheels.
- If you bundle CUDA-enabled PyTorch, NeMo, CUDA, or NVIDIA runtime packages, include the applicable
  NVIDIA, PyTorch, CUDA, and package-specific notices for the exact wheels shipped.
- If you bundle FFmpeg binaries, include the license and configuration notices for the exact FFmpeg
  build shipped. Segmentwright itself does not vendor FFmpeg.
- If you bundle Ollama or an Ollama-managed model cache, include the notices and terms for Ollama and
  each bundled model.
- If you redistribute FireRedVAD or NVIDIA Parakeet model weights, preserve the applicable upstream
  model-card license and attribution material.
- If you redistribute Gemma model weights, review and comply with Google's Gemma terms for the exact
  model artifact.
- If you enable paid Gemini transcription for users, review and comply with the applicable Google API
  and model terms for your product context.
- Refresh this notice whenever `pyproject.toml`, `uv.lock`, bundled runtimes, model defaults, or
  model download sources change in a way that affects third-party obligations.
