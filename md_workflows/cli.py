"""Typer CLI: a subcommand per step plus ``run-pipeline``.

The CLI is a thin caller of ``core`` (and the SDK): it resolves ``--workdir`` +
``--config`` (+ flag overrides) into typed inputs, runs the step, and maps structured
exceptions to process exit codes. It never imports orchestration code.

Exit codes: 0 ok · 1 domain error · 2 missing input · 3 external tool failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from .core.config import RUN_PROFILE_KEYS, SystemConfig
from .core.exceptions import MDWorkflowError, MissingInputError, StepToolError
from .core.pipelines.standard_md import standard_md_pipeline
from .core.results import StepResult, StepStatus
from .core.steps import (
    equilibrate as equilibrate_step,
)
from .core.steps import (
    make_crystal as make_crystal_step,
)
from .core.steps import (
    make_waterbox as make_waterbox_step,
)
from .core.steps import (
    minimize as minimize_step,
)
from .core.steps import (
    param_prot as param_prot_step,
)
from .core.steps import (
    resolvate as resolvate_step,
)
from .core.steps import (
    run_params_gaussian as gaussian_step,
)
from .core.steps import (
    solvate as solvate_step,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Crystalline molecular-dynamics prep workflows.",
)


@dataclass
class Common:
    workdir: Path
    config: Path | None
    resume: bool


@app.callback()
def _main(
    ctx: typer.Context,
    workdir: Path = typer.Option(Path("."), "--workdir", "-w", help="Run directory."),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML/TOML config file."),
    resume: bool = typer.Option(
        False, "--resume/--force", help="Skip steps whose outputs already exist."
    ),
) -> None:
    ctx.obj = Common(workdir=workdir, config=config, resume=resume)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_cfg(common: Common, overrides: dict[str, Any] | None = None) -> SystemConfig:
    try:
        return SystemConfig.load(common.config, overrides=overrides or {})
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


def _ntomp_override(ntomp: int | None) -> dict[str, Any]:
    """Apply a quick ntomp tweak across every mdrun profile."""
    if ntomp is None:
        return {}
    return {"run_profiles": {k: {"ntomp": ntomp} for k in RUN_PROFILE_KEYS}}


def _parse_op_order(value: str | None) -> list[int] | None:
    """Parse a comma/space separated 1-based symmetry-operation order."""
    if value is None:
        return None
    try:
        return [int(token) for token in value.replace(",", " ").split()]
    except ValueError as exc:
        typer.secho(
            f"invalid --op-order {value!r}: expected integers", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1) from exc


def _emit(result: StepResult) -> None:
    color = {
        StepStatus.COMPLETED: typer.colors.GREEN,
        StepStatus.SKIPPED: typer.colors.YELLOW,
        StepStatus.FAILED: typer.colors.RED,
    }[result.status]
    typer.secho(f"{result.step}: {result.status.value}", fg=color)
    for name, path in result.outputs.items():
        typer.echo(f"  {name}: {path}")


def _run(fn) -> None:
    """Execute a step callable, mapping structured exceptions to exit codes."""
    try:
        result = fn()
    except MissingInputError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    except StepToolError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(3) from exc
    except MDWorkflowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    if isinstance(result, StepResult):
        _emit(result)
    else:  # pipeline result
        for step in result.steps:
            _emit(step)


# --------------------------------------------------------------------------- #
# per-step commands
# --------------------------------------------------------------------------- #
@app.command("param-prot")
def param_prot_cmd(
    ctx: typer.Context,
    pdb_id: str | None = typer.Option(None, "--pdb-id", help="PDB id to parameterize."),
) -> None:
    """Parameterize the protein (clean, tleap, Amber->GROMACS)."""
    common: Common = ctx.obj
    cfg = _load_cfg(common, {"pdb_id": pdb_id} if pdb_id else None)
    inputs = param_prot_step.resolve_inputs(common.workdir, cfg)
    _run(lambda: param_prot_step.param_prot(inputs, resume=common.resume))


@app.command("make-crystal")
def make_crystal_cmd(
    ctx: typer.Context,
    ix: int | None = typer.Option(None, "--ix", help="Supercell replication in x (and y/z)."),
    iy: int | None = typer.Option(None, "--iy"),
    iz: int | None = typer.Option(None, "--iz"),
    spacegroup: str | None = typer.Option(
        None, "--spacegroup", help="Override the space group in CRYST1, e.g. 'P 21 21 21'."
    ),
    op_order: str | None = typer.Option(
        None,
        "--op-order",
        help="1-based symmetry-operation order, e.g. '1,2,4,3'; reorders the cell's chains.",
    ),
    numbering: str | None = typer.Option(
        None, "--numbering", help="Supercell numbering: auto (default), continuous or per-cell."
    ),
    chimerax_exec: str | None = typer.Option(
        None, "--chimerax-exec", help="Unused: the expansion is gemmi-based. Kept for callers."
    ),
) -> None:
    """Build the crystal supercell."""
    common: Common = ctx.obj
    crystal = {
        k: v
        for k, v in {
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "spacegroup": spacegroup,
            "op_order": _parse_op_order(op_order),
            "numbering": numbering,
            "chimerax_exec": chimerax_exec,
        }.items()
        if v is not None
    }
    cfg = _load_cfg(common, {"crystal": crystal} if crystal else None)
    inputs = make_crystal_step.resolve_inputs(common.workdir, cfg)
    _run(lambda: make_crystal_step.make_crystal(inputs, cfg.crystal, resume=common.resume))


@app.command("make-waterbox")
def make_waterbox_cmd(
    ctx: typer.Context,
    nc_scale: int | None = typer.Option(None, "--nc-scale"),
    conc: float | None = typer.Option(None, "--conc"),
    ntomp: int | None = typer.Option(None, "--ntomp", help="Override mdrun OpenMP threads."),
) -> None:
    """Build the equilibrated bulk-water reservoir."""
    common: Common = ctx.obj
    wb = {k: v for k, v in {"nc_scale": nc_scale, "conc": conc}.items() if v is not None}
    overrides = _ntomp_override(ntomp)
    if wb:
        overrides["waterbox"] = wb
    cfg = _load_cfg(common, overrides)
    inputs = make_waterbox_step.resolve_inputs(common.workdir, cfg)
    _run(
        lambda: make_waterbox_step.make_waterbox(
            inputs,
            cfg.waterbox,
            cfg.profile("waterbox_min"),
            cfg.profile("waterbox_equil"),
            resume=common.resume,
        )
    )


@app.command("solvate")
def solvate_cmd(
    ctx: typer.Context,
    ionic_strength: float | None = typer.Option(None, "--ionic-strength", help="Target salt M."),
) -> None:
    """Solvate the crystal and add ions."""
    common: Common = ctx.obj
    ov = {"solvate": {"ionic_strength": ionic_strength}} if ionic_strength is not None else None
    cfg = _load_cfg(common, ov)
    inputs = solvate_step.resolve_inputs(common.workdir, cfg)
    _run(lambda: solvate_step.solvate(inputs, cfg.solvate, resume=common.resume))


@app.command("minimize")
def minimize_cmd(
    ctx: typer.Context,
    ntomp: int | None = typer.Option(None, "--ntomp", help="Override mdrun OpenMP threads."),
) -> None:
    """Energy-minimize the solvated model."""
    common: Common = ctx.obj
    cfg = _load_cfg(common, _ntomp_override(ntomp))
    inputs = minimize_step.resolve_inputs(common.workdir, cfg)
    _run(lambda: minimize_step.minimize(inputs, cfg.profile("min"), resume=common.resume))


@app.command("equilibrate")
def equilibrate_cmd(
    ctx: typer.Context,
    ntomp: int | None = typer.Option(None, "--ntomp"),
    ligand_resname: str | None = typer.Option(None, "--ligand-resname"),
    handle_ligand_restraint: bool | None = typer.Option(
        None, "--handle-ligand-restraint/--no-ligand-restraint"
    ),
    chain_split_mode: str | None = typer.Option(
        None, "--chain-split-mode", help="ignore_ligand | split_intelligently"
    ),
) -> None:
    """Set up restraints and run the NPT equilibration."""
    common: Common = ctx.obj
    eq = {
        k: v
        for k, v in {
            "ligand_resname": ligand_resname,
            "handle_ligand_restraint": handle_ligand_restraint,
            "chain_split_mode": chain_split_mode,
        }.items()
        if v is not None
    }
    overrides = _ntomp_override(ntomp)
    if eq:
        overrides["equilibrate"] = eq
    cfg = _load_cfg(common, overrides)
    inputs = equilibrate_step.resolve_inputs(common.workdir, cfg)
    _run(
        lambda: equilibrate_step.equilibrate(
            inputs, cfg.equilibrate, cfg.profile("equil"), resume=common.resume
        )
    )


@app.command("resolvate")
def resolvate_cmd(
    ctx: typer.Context,
    target_bar: float | None = typer.Option(None, "--target-bar", help="Target mean pressure."),
    trial_fraction: float | None = typer.Option(None, "--trial-fraction"),
    ntomp: int | None = typer.Option(None, "--ntomp"),
) -> None:
    """Adaptive resolvation to the target pressure (run 1 -> branch -> final)."""
    common: Common = ctx.obj
    rv = {
        k: v
        for k, v in {"target_bar": target_bar, "trial_fraction": trial_fraction}.items()
        if v is not None
    }
    overrides = _ntomp_override(ntomp)
    if rv:
        overrides["resolvate"] = rv
    cfg = _load_cfg(common, overrides)
    inputs = resolvate_step.resolve_inputs(common.workdir, cfg)
    _run(
        lambda: resolvate_step.resolvate(
            inputs,
            cfg.resolvate,
            cfg.profile("resolv_min"),
            cfg.profile("resolv_equil"),
            resume=common.resume,
        )
    )


@app.command("run-params-gaussian")
def run_params_gaussian_cmd(
    ctx: typer.Context,
    pdb_id: str | None = typer.Option(None, "--pdb-id"),
    g16root: str | None = typer.Option(None, "--g16root"),
    nproc: int | None = typer.Option(None, "--nproc"),
) -> None:
    """Ligand parameterization with Gaussian + AmberTools (standalone)."""
    common: Common = ctx.obj
    gaussian = {k: v for k, v in {"g16root": g16root, "nproc": nproc}.items() if v is not None}
    overrides: dict[str, Any] = {}
    if pdb_id:
        overrides["pdb_id"] = pdb_id
    if gaussian:
        overrides["gaussian"] = gaussian
    cfg = _load_cfg(common, overrides)
    inputs = gaussian_step.resolve_inputs(common.workdir, cfg)
    _run(lambda: gaussian_step.run_params_gaussian(inputs, cfg.gaussian, resume=common.resume))


@app.command("run-pipeline")
def run_pipeline_cmd(
    ctx: typer.Context,
    pdb_id: str | None = typer.Option(None, "--pdb-id"),
    ix: int | None = typer.Option(None, "--ix", help="Crystal supercell replication."),
    ntomp: int | None = typer.Option(None, "--ntomp", help="Override mdrun OpenMP threads."),
) -> None:
    """Run the full prep pipeline (param_prot -> ... -> resolvate)."""
    common: Common = ctx.obj
    overrides = _ntomp_override(ntomp)
    if pdb_id:
        overrides["pdb_id"] = pdb_id
    if ix is not None:
        overrides["crystal"] = {"ix": ix}
    cfg = _load_cfg(common, overrides)
    _run(lambda: standard_md_pipeline(common.workdir, cfg, resume=common.resume))


if __name__ == "__main__":
    app()
