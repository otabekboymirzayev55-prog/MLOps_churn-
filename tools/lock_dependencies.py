"""Record installed runtime/test dependency versions, including their dependencies."""
from importlib.metadata import distribution
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
pending = [Requirement(line) for line in (root / "requirements.txt").read_text().splitlines() if line.strip()]
resolved = {}
while pending:
    requirement = pending.pop()
    key = canonicalize_name(requirement.name)
    if key in resolved:
        continue
    package = distribution(requirement.name)
    resolved[key] = package.version
    for dependency in package.requires or []:
        child = Requirement(dependency)
        if child.marker is None or any(child.marker.evaluate({"extra": extra}) for extra in {"", *requirement.extras}):
            pending.append(child)
# Windows-only dependencies must not be installed in the Linux image.
lines = [f"{name}=={version}" + ('; sys_platform == "win32"' if name in {"pywin32", "pyreadline3"} else "") for name, version in sorted(resolved.items())]
(root / "requirements.lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
