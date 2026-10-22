from dataclasses import replace
from pathlib import Path

import pytest

import segmentwright.cli as cli_module
from segmentwright.cli import build_config, build_parser, main
from segmentwright.config import resolve_alignment_model, validate_config


def test_cli_defaults(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args([str(input_path), str(output_path)])
    config = build_config(args)

    assert config.env_file == Path("~/.segmentwright/.env").expanduser()
    assert config.preset == "quality-baseline"
    assert config.language == "ja"
    assert config.gemini_model == "gemini-3.8-flash"
    assert config.alignment_model is None
    assert config.gemini_thinking_level is None
    assert config.gemini_temperature is None
    assert config.prompt_preset == "subtitle-lines"
    assert config.response_format == "json_lines"
    assert config.segmentation == "direct"
    assert config.resegmentation_model == "gemma4:12b"
    assert config.resegmentation_prompt == "medium"
    assert config.ollama_auto_pull is False
    assert config.ollama_start_timeout_sec == 30.0
    assert config.artifacts_explicit is False
    assert config.firered_vad_auto_download is True
    assert config.target_chunk_sec == 180
    assert config.max_chunk_sec == 180
    assert config.max_retry_work_items is None
    assert config.cache_enabled is True


def test_cli_accepts_explicit_retry_budget(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--max-retry-work-items", "20"]
    )

    assert build_config(args).max_retry_work_items == 20


def test_cli_rejects_removed_max_work_items_option(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    with pytest.raises(SystemExit):
        build_parser().parse_args([str(input_path), str(output_path), "--max-work-items", "20"])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_validate_rejects_nonpositive_retry_budget(tmp_path: Path, value: str) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    config = build_config(
        build_parser().parse_args(
            [str(input_path), str(output_path), "--max-retry-work-items", value]
        )
    )

    with pytest.raises(ValueError, match="max-retry-work-items"):
        validate_config(config)


def test_cli_accepts_english_language(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args([str(input_path), str(output_path), "--language", "en"])
    config = build_config(args)

    assert config.language == "en"
    assert config.language_explicit is True


def test_cli_accepts_transcript_import_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text('{"version":1,"language":"ja","chunks":[]}', encoding="utf-8")

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--transcript", str(transcript_path)]
    )
    config = build_config(args)

    assert config.transcript_path == transcript_path
    assert config.segmentation == "resegmentation"
    assert config.language == "ja"
    assert config.language_explicit is False


def test_validate_rejects_missing_transcript_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    config = build_config(
        build_parser().parse_args(
            [str(input_path), str(output_path), "--transcript", str(tmp_path / "missing.json")]
        )
    )

    with pytest.raises(ValueError, match="Transcript file does not exist"):
        validate_config(config)


def test_validate_rejects_gemini_options_with_transcript(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text('{"version":1,"language":"ja","chunks":[]}', encoding="utf-8")

    config = build_config(
        build_parser().parse_args(
            [
                str(input_path),
                str(output_path),
                "--transcript",
                str(transcript_path),
                "--temperature",
                "0.0",
            ]
        )
    )

    with pytest.raises(ValueError, match="temperature"):
        validate_config(config)


def test_cli_accepts_alignment_model_override(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--alignment-model", "custom-aligner"]
    )
    config = build_config(args)

    assert config.alignment_model == "custom-aligner"


def test_alignment_model_defaults_by_language(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    ja_config = build_config(build_parser().parse_args([str(input_path), str(output_path)]))
    en_config = build_config(
        build_parser().parse_args([str(input_path), str(output_path), "--language", "en"])
    )

    assert resolve_alignment_model(ja_config) == "nvidia/parakeet-tdt_ctc-0.6b-ja"
    assert resolve_alignment_model(en_config) == "nvidia/parakeet-ctc-0.6b"


def test_cli_rejects_removed_cheaper_candidate_preset(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [str(input_path), str(output_path), "--preset", "cheaper-candidate"]
        )


@pytest.mark.parametrize("disable", [False, True])
def test_cli_resegmentation_preserves_line_breaks_by_default(tmp_path: Path, disable: bool) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    arguments = [str(input_path), str(tmp_path / "out.srt")]
    if disable:
        arguments.append("--no-resegmentation-preserve-line-breaks")
    config = build_config(build_parser().parse_args(arguments))

    assert config.resegmentation_preserve_line_breaks is not disable


def test_cli_lite_experimental_preset(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--preset", "lite-experimental"]
    )
    config = build_config(args)

    assert config.preset == "lite-experimental"
    assert config.gemini_model == "gemini-3.5-flash-lite"
    assert config.prompt_preset == "plain-transcript"
    assert config.response_format == "plain_text"
    assert config.segmentation == "resegmentation"
    assert config.resegmentation_prompt == "medium"


def test_cli_manual_overrides_preserve_unset_values(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [
            str(input_path),
            str(output_path),
            "--preset",
            "quality-baseline",
            "--gemini-model",
            "custom-model",
            "--prompt-preset",
            "subtitle-lines",
            "--response-format",
            "json_lines",
            "--segmentation",
            "direct",
            "--resegmentation-prompt",
            "short",
            "--ollama-model",
            "gemma4:12b",
            "--ollama-auto-pull",
            "--ollama-start-timeout-sec",
            "5",
        ]
    )
    config = build_config(args)

    assert config.preset == "quality-baseline"
    assert config.gemini_model == "custom-model"
    assert config.prompt_preset == "subtitle-lines"
    assert config.response_format == "json_lines"
    assert config.segmentation == "direct"
    assert config.resegmentation_prompt == "short"
    assert config.resegmentation_model == "gemma4:12b"
    assert config.ollama_auto_pull is True
    assert config.ollama_start_timeout_sec == 5.0


def test_cli_accepts_explicit_artifacts_dir(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    artifacts_path = tmp_path / "debug-artifacts"

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--artifacts", str(artifacts_path)]
    )
    config = build_config(args)

    assert config.artifacts_explicit is True
    assert config.artifacts_dir == artifacts_path


def test_cli_rejects_removed_keep_temp_flag(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    with pytest.raises(SystemExit):
        build_parser().parse_args([str(input_path), str(output_path), "--keep-temp"])


def test_cli_accepts_prompt_file_and_text(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("custom prompt", encoding="utf-8")

    file_args = build_parser().parse_args(
        [str(input_path), str(output_path), "--prompt-file", str(prompt_file)]
    )
    text_args = build_parser().parse_args(
        [str(input_path), str(output_path), "--prompt-text", "custom prompt"]
    )

    assert build_config(file_args).prompt_file == prompt_file
    assert build_config(text_args).prompt_text == "custom prompt"


def test_validate_rejects_prompt_file_and_text_together(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("custom prompt", encoding="utf-8")

    config = build_config(
        build_parser().parse_args(
            [
                str(input_path),
                str(output_path),
                "--prompt-file",
                str(prompt_file),
                "--prompt-text",
                "custom prompt",
            ]
        )
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(config)


def test_validate_rejects_missing_prompt_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    config = build_config(
        build_parser().parse_args(
            [str(input_path), str(output_path), "--prompt-file", str(tmp_path / "missing.txt")]
        )
    )

    with pytest.raises(ValueError, match="Prompt file does not exist"):
        validate_config(config)


def test_validate_rejects_output_pointing_to_input_file(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    output_path.symlink_to(input_path)

    config = build_config(
        build_parser().parse_args(
            [str(input_path), str(output_path), "--overwrite"]
        )
    )

    with pytest.raises(ValueError, match="must not point to the input file"):
        validate_config(config)


def test_cli_can_disable_firered_vad_auto_download(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--no-firered-vad-auto-download"]
    )
    config = build_config(args)

    assert config.firered_vad_auto_download is False


def test_cli_accepts_gemini_thinking_level(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [str(input_path), str(output_path), "--gemini-thinking-level", "low"]
    )
    config = build_config(args)

    assert config.gemini_thinking_level == "low"


def test_cli_accepts_gemini_temperature(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args([str(input_path), str(output_path), "--temperature", "0.0"])
    config = build_config(args)

    assert config.gemini_temperature == 0.0


def test_validate_rejects_invalid_gemini_temperature(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args([str(input_path), str(output_path), "--temperature", "-0.1"])
    config = build_config(args)

    with pytest.raises(ValueError, match="temperature"):
        validate_config(config)


def test_validate_rejects_invalid_resegmentation_prompt(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    config = build_config(build_parser().parse_args([str(input_path), str(output_path)]))
    config = replace(config, resegmentation_prompt="experimental")

    with pytest.raises(ValueError, match="Invalid resegmentation prompt"):
        validate_config(config)


def test_validate_rejects_invalid_ollama_config(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    config = build_config(build_parser().parse_args([str(input_path), str(output_path)]))

    with pytest.raises(ValueError, match="ollama-start-timeout-sec"):
        validate_config(replace(config, ollama_start_timeout_sec=0))


def test_validate_rejects_target_above_hard_max(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"

    args = build_parser().parse_args(
        [
            str(input_path),
            str(output_path),
            "--target-chunk-sec",
            "301",
            "--max-chunk-sec",
            "300",
        ]
    )
    config = build_config(args)

    with pytest.raises(ValueError, match="must not exceed"):
        validate_config(config)


def test_validate_ignores_nonempty_artifacts_dir_when_not_explicit(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    artifacts_path = tmp_path / "nonempty"
    artifacts_path.mkdir()
    (artifacts_path / "old.txt").write_text("old", encoding="utf-8")

    config = build_config(build_parser().parse_args([str(input_path), str(output_path)]))
    validate_config(replace(config, artifacts_dir=artifacts_path, artifacts_explicit=False))

    with pytest.raises(ValueError, match="Artifacts directory is not empty"):
        validate_config(replace(config, artifacts_dir=artifacts_path, artifacts_explicit=True))


def test_main_uses_temporary_artifacts_dir_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    seen: dict[str, Path | bool] = {}

    def fake_run_pipeline(config) -> None:
        seen["artifacts_dir"] = config.artifacts_dir
        seen["artifacts_explicit"] = config.artifacts_explicit
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (config.artifacts_dir / "during-run.txt").write_text("temporary", encoding="utf-8")
        config.output_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_module, "cleanup_stale_temp_workdirs", lambda: [])
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    assert main([str(input_path), str(output_path)]) == 0

    artifacts_dir = seen["artifacts_dir"]
    assert isinstance(artifacts_dir, Path)
    assert artifacts_dir.name.startswith("segmentwright-")
    assert artifacts_dir.exists() is False
    assert seen["artifacts_explicit"] is False


def test_main_preserves_explicit_artifacts_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    artifacts_path = tmp_path / "artifacts"

    def fake_run_pipeline(config) -> None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (config.artifacts_dir / "preserved.txt").write_text("preserved", encoding="utf-8")
        config.output_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_module, "cleanup_stale_temp_workdirs", lambda: [])
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    assert main([str(input_path), str(output_path), "--artifacts", str(artifacts_path)]) == 0
    assert (artifacts_path / "preserved.txt").read_text(encoding="utf-8") == "preserved"


def test_main_removes_temporary_artifacts_dir_after_reported_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    seen: dict[str, Path] = {}

    def fake_run_pipeline(config) -> None:
        seen["artifacts_dir"] = config.artifacts_dir
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (config.artifacts_dir / "errors.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("reported failure")

    monkeypatch.setattr(cli_module, "cleanup_stale_temp_workdirs", lambda: [])
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    assert main([str(input_path), str(output_path)]) == 1

    artifacts_dir = seen["artifacts_dir"]
    assert artifacts_dir.exists() is False


def test_main_preserves_explicit_artifacts_dir_after_reported_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    output_path = tmp_path / "output.srt"
    artifacts_path = tmp_path / "artifacts"

    def fake_run_pipeline(config) -> None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (config.artifacts_dir / "errors.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("reported failure")

    monkeypatch.setattr(cli_module, "cleanup_stale_temp_workdirs", lambda: [])
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    assert (
        main([str(input_path), str(output_path), "--artifacts", str(artifacts_path)])
        == 1
    )
    assert (artifacts_path / "errors.json").is_file()
