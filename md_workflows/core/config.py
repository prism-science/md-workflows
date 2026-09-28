"""Typed configuration for the MD prep pipeline.

A :class:`SystemConfig` captures everything that varies between systems and machines:
the PDB id, crystal/water-box/solvation parameters, ligand-handling flags, and — the
dominant cross-machine knob — the per-invocation GROMACS ``mdrun`` execution profiles
(GPU offload + thread counts). It loads from YAML/TOML and accepts CLI-flag overrides.

Defaults follow the canonical taylor scripts
(``taylor_scripts_4lzt_5x5x5``): GPU ``mdrun`` with ``-pme cpu`` on minimizations and
``-pme gpu`` on equilibrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import tomllib
import yaml
from pydantic import BaseModel, ConfigDict, Field

Offload = Literal["auto", "cpu", "gpu"]

# Atom/residue numbering of a propagated supercell; see ``core.crystal``.
CrystalNumbering = Literal["auto", "continuous", "per-cell"]

# Keys for the per-invocation mdrun profiles. Each GROMACS mdrun call in the pipeline
# looks up its profile by one of these names.
RUN_PROFILE_KEYS = (
    "waterbox_min",
    "waterbox_equil",
    "min",
    "equil",
    "resolv_min",
    "resolv_equil",
)


class GromacsRunProfile(BaseModel):
    """Execution flags for a single ``gmx mdrun`` invocation.

    These are hardware-dependent and differ per call (e.g. ``-pme cpu`` for
    minimization vs ``-pme gpu`` for equilibration), so they are configured rather than
    hard-coded. ``to_mdrun_flags`` renders only the set (non-``None``) options.
    """

    model_config = ConfigDict(extra="forbid")

    gmx_bin: str = "gmx"
    ntmpi: int | None = 1
    ntomp: int | None = 16
    nb: Offload | None = "gpu"
    pme: Offload | None = "gpu"
    bonded: Offload | None = "gpu"
    update: Offload | None = None
    pin: Literal["auto", "on", "off"] | None = None
    dlb: Literal["auto", "yes", "no"] | None = None
    tunepme: bool = True  # emit -notunepme when False
    extra_mdrun_args: list[str] = Field(default_factory=list)

    def to_mdrun_flags(self) -> list[str]:
        """Render the mdrun option list (excludes ``-s``/``-deffnm``/``-v``)."""
        flags: list[str] = []
        if self.ntmpi is not None:
            flags += ["-ntmpi", str(self.ntmpi)]
        if self.ntomp is not None:
            flags += ["-ntomp", str(self.ntomp)]
        if self.nb is not None:
            flags += ["-nb", self.nb]
        if self.pme is not None:
            flags += ["-pme", self.pme]
        if self.bonded is not None:
            flags += ["-bonded", self.bonded]
        if self.update is not None:
            flags += ["-update", self.update]
        if self.pin is not None:
            flags += ["-pin", self.pin]
        if self.dlb is not None:
            flags += ["-dlb", self.dlb]
        if not self.tunepme:
            flags += ["-notunepme"]
        flags += list(self.extra_mdrun_args)
        return flags


def _min_profile() -> GromacsRunProfile:
    # Minimizations: GPU nonbonded, CPU PME/bonded, no PME tuning (taylor canonical).
    return GromacsRunProfile(nb="gpu", pme="cpu", bonded="cpu", tunepme=False)


def _equil_profile() -> GromacsRunProfile:
    # Equilibrations: full GPU offload, no PME tuning (taylor canonical).
    return GromacsRunProfile(nb="gpu", pme="gpu", bonded="gpu", tunepme=False)


def _default_run_profiles() -> dict[str, GromacsRunProfile]:
    return {
        "waterbox_min": _min_profile(),
        "waterbox_equil": _min_profile(),  # taylor equilibrates the water box on CPU PME
        "min": _min_profile(),
        "equil": _equil_profile(),
        "resolv_min": _min_profile(),
        "resolv_equil": _equil_profile(),
    }


class CrystalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ix: int = 1
    iy: int | None = None  # falls back to ix
    iz: int | None = None  # falls back to ix
    spacegroup: str | None = None  # None => take it from the CRYST1 record
    # 1-based reordering of the space group's symmetry operations, e.g. [1, 2, 4, 3] to
    # reproduce the chain labelling of the old ChimeraX route; see ``core.crystal``.
    op_order: list[int] | None = None
    numbering: CrystalNumbering = "auto"
    # Retained so existing configs and --chimerax-exec keep validating; the expansion is
    # gemmi-based now and never launches ChimeraX.
    chimerax_exec: str = "chimerax"


class WaterboxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nc_scale: int = 5  # cell subdivision / reservoir tiling factor (taylor)
    conc: float = 60.0  # gmx insert-molecules water concentration (mol/L)
    # Numbering for the nc_scale^3 tiling; see ``core.crystal.propagate_cell``. "auto"
    # keeps PropPDB-compatible continuous numbering until the water count overruns the
    # PDB resSeq field, then restarts the counters in each tile.
    numbering: CrystalNumbering = "auto"
    min_mdp: str = "min_water.mdp"
    equil_mdp: str = "equil_water.mdp"


class SolvateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ionic_strength: float = 0.1  # target molarity for added ions
    water_molarity: float = 55.0  # water molarity used in the ion-count formula


class MinimizeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_mdp: str = "min.mdp"


class EquilibrateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equil_mdp: str = "equil.mdp"
    restraint_fc: float = 209.2  # kJ/mol/nm^2 position-restraint force constant
    restraint_group: str = "Protein-H"  # gmx genrestr selection group
    ligand_resname: str | None = None  # None => auto-detect via pdb_file_processing
    handle_ligand_restraint: bool = False  # OFF until the full ligand feature lands
    chain_split_mode: Literal["ignore_ligand", "split_intelligently"] = "ignore_ligand"


class ResolvateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_fraction: float = 0.25  # fraction of trial-solvation waters used in run 1
    scale_add: float = 1.5  # run-2 probe multiplier on run-1 water count
    target_bar: float = 1.0  # target mean pressure
    pressure_tol: float = 100.0  # bar; within this of target counts as "converged"
    max_dNw_factor: float = 1.0  # sanity bound on |dNw| as a multiple of current waters


class GaussianParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    g16root: str = ""  # required only when running the standalone gaussian utility
    nproc: int = 8
    net_charge: int = -2
    method: str = "B3LYP/6-31+G(d,p)"


class SystemConfig(BaseModel):
    """Top-level configuration for one MD prep run."""

    model_config = ConfigDict(extra="forbid")

    pdb_id: str = "6B8X"
    variant: Literal["original", "taylor"] = "taylor"
    mdp_dir: Path = Path("artifacts")

    crystal: CrystalParams = Field(default_factory=CrystalParams)
    waterbox: WaterboxParams = Field(default_factory=WaterboxParams)
    solvate: SolvateParams = Field(default_factory=SolvateParams)
    minimize: MinimizeParams = Field(default_factory=MinimizeParams)
    equilibrate: EquilibrateParams = Field(default_factory=EquilibrateParams)
    resolvate: ResolvateParams = Field(default_factory=ResolvateParams)
    gaussian: GaussianParams = Field(default_factory=GaussianParams)

    run_profiles: dict[str, GromacsRunProfile] = Field(default_factory=_default_run_profiles)

    def profile(self, key: str) -> GromacsRunProfile:
        """Return the mdrun profile for an invocation key, defaulting if unset."""
        return self.run_profiles.get(key, GromacsRunProfile())

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> SystemConfig:
        """Build a config from an optional YAML/TOML file plus optional overrides.

        ``overrides`` is a (possibly nested) dict deep-merged over the file/defaults —
        this is how CLI flags win over the config file.

        The merge is seeded from the full default config so a partial override (e.g.
        one field of one run profile) merges into the defaults instead of replacing the
        whole ``run_profiles`` mapping.
        """
        data: dict[str, Any] = cls().model_dump(mode="python")
        if path is not None:
            data = _deep_merge(data, _read_config_file(Path(path)))
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.model_validate(data)


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(path.read_text()) or {}
        elif suffix == ".toml":
            loaded = tomllib.loads(path.read_text())
        else:
            raise ValueError(f"unsupported config format {suffix!r} (use .yaml/.yml/.toml)")
    except (yaml.YAMLError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"could not parse config file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} must contain a mapping at the top level")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins), non-mutating."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged
