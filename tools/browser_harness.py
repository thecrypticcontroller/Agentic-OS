from __future__ import annotations

import shutil
import subprocess
from typing import Final


HARNESS_FALLBACK: Final[str] = r"D:\AI\uv\bin\browser-harness.exe"


def run_browser_script(script: str) -> str:
    executable = shutil.which("browser-harness") or HARNESS_FALLBACK

    result = subprocess.run(
        [executable],
        input=script,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"browser-harness failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()
