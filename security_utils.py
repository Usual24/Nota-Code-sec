from pathlib import Path


def sanitize_repo_relative_path(path: str) -> Path:
    candidate = Path(path.strip())
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError("저장소 상대경로만 허용됩니다.")
    normalized = Path(*[part for part in candidate.parts if part not in {"", "."}])
    if not normalized.parts:
        raise ValueError("유효한 경로를 입력해 주세요.")
    return normalized
