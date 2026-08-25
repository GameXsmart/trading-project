"""Watch cloudflared's log and print the public address it was assigned.

Separated from ``share.bat`` because extracting a URL from a log file is string
work, and batch is a poor language for string work — the first version of this
lived as a one-line Python expression embedded in the batch file, which was
unreadable and therefore unmaintainable.

Exits 0 once an address appears, 1 if none appears before the timeout. The batch
file reads that exit code rather than trying to parse this output.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

# The assigned hostname is several words joined by hyphens. Anchored to the
# hyphenated form on purpose: the log also mentions `api.trycloudflare.com`,
# which is Cloudflare's own endpoint and not a tunnel — matching it produced a
# confident, wrong answer the first time this was written.
_ADDRESS = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+){2,}\.trycloudflare\.com")

_TIMEOUT_SECONDS = 90
_POLL_SECONDS = 1.0


def find_address(log: Path) -> str | None:
    if not log.exists():
        return None
    match = _ADDRESS.search(log.read_text(encoding="utf-8", errors="ignore"))
    return match.group(0) if match else None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wait_for_tunnel.py <cloudflared-log-path>")
        return 2
    log = Path(sys.argv[1])

    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        address = find_address(log)
        if address:
            print()
            print("  " + "=" * 58)
            print()
            print(f"    PUBLIC LINK   {address}")
            print()
            print("  " + "=" * 58)
            return 0
        time.sleep(_POLL_SECONDS)

    print()
    print("  Cloudflare did not assign an address within "
          f"{_TIMEOUT_SECONDS} seconds.")
    print("  This usually means a slow network; running share.bat again")
    print("  normally works. The full log is at:")
    print(f"    {log}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
