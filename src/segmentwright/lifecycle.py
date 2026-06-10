from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelLifecycleRecorder:
    output_path: Path
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        stage_id: str,
        generation: int | None,
        role: str,
        backend: str,
        model_name: str | None,
        device: str | None,
        event: str,
        local: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "stage_id": stage_id,
                "generation": generation,
                "role": role,
                "backend": backend,
                "model_name": model_name,
                "device": device,
                "event": event,
                "local": local,
                "pid": os.getpid(),
                "monotonic_sec": time.monotonic(),
                "wall_time_sec": time.time(),
                "details": details or {},
            }
        )

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "segmentwright.model_lifecycle.v1",
            "events": self.events,
        }
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
