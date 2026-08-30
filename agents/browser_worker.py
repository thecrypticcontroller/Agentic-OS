from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.browser_harness import run_browser_script


def open_and_inspect(url: str) -> str:
    script = f"""
new_tab({url!r})
wait_for_load()
print(page_info())
"""

    return run_browser_script(script)


if __name__ == "__main__":
    result = open_and_inspect("https://example.com")
    print("Browser Worker result:")
    print(result)
