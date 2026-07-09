from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DIRS: tuple[str, ...] = (
    "configs",
    "data/raw/events",
    "data/raw/ntl/vnp46a2_daily_or_monthly_ingest",
    "data/raw/rainfall/gpm_imerg",
    "data/raw/flood/sentinel1_flood",
    "data/raw/flood/global_flood_database",
    "data/raw/socioeconomic/population",
    "data/raw/socioeconomic/builtup",
    "data/raw/socioeconomic/roads",
    "data/raw/socioeconomic/electrification",
    "data/raw/boundaries/india_admin_districts",
    "data/raw/boundaries/india_states",
    "data/raw/boundaries/india_coastline_rivers",
    "data/interim/grid",
    "data/interim/events",
    "data/interim/ntl",
    "data/interim/features",
    "data/interim/targets",
    "data/processed/splits",
    "data/processed/metadata",
    "data/external",
    "outputs/models",
    "outputs/predictions",
    "outputs/metrics",
    "outputs/tables",
    "outputs/figures",
    "outputs/maps",
    "outputs/final_bundle",
    "notebooks",
    "scripts",
    "src/ntlpol/data",
    "src/ntlpol/extractors",
    "src/ntlpol/targets",
    "src/ntlpol/models",
    "src/ntlpol/evaluation",
    "src/ntlpol/visualization",
)


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project paths.

    The class keeps all scripts using the same directory convention.
    It intentionally does not assume that raw data already exists.
    """

    root: Path

    @classmethod
    def from_current_file(cls, file: str | Path, levels_up: int = 1) -> "ProjectPaths":
        root = Path(file).resolve()
        for _ in range(levels_up):
            root = root.parent
        return cls(root=root)

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(root=Path(root).resolve())

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def metadata(self) -> Path:
        return self.processed / "metadata"

    def resolve(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.root / path

    def ensure_dirs(self, extra_dirs: list[str] | None = None, keep: bool = True) -> None:
        dirs = list(DEFAULT_DIRS)
        if extra_dirs:
            dirs.extend(extra_dirs)
        for rel in dirs:
            path = self.root / rel
            path.mkdir(parents=True, exist_ok=True)
            if keep:
                keep_file = path / ".gitkeep"
                keep_file.touch(exist_ok=True)
