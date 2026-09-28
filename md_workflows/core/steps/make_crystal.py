"""Build a crystal supercell from the parameterized protein.

Corresponds to ``make_crystal.sh``: dry the protein, restore the CRYST1 cell, expand
the asymmetric unit to the full unit cell in space group P1, then replicate it into a
supercell. The expansion and the replication are done with gemmi (``core.crystal``),
which replaces the ChimeraX ``unitcell`` and AmberTools ``PropPDB`` calls the shell
recipe shelled out to; ``pdb4amber`` is the only external tool this step still needs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import CrystalParams, SystemConfig
from ..gmx import checksums
from ..results import StepInputs, StepResult, StepStatus

STEP = "make_crystal"


class MakeCrystalInputs(StepInputs):
    prot_pdb: Path
    pdb_clean: Path

    def consumed_paths(self) -> list[Path]:
        return [self.prot_pdb, self.pdb_clean]


class MakeCrystalResult(StepResult):
    @property
    def xtal(self) -> Path:
        return self.output("xtal")


def resolve_inputs(workdir: Path, cfg: SystemConfig) -> MakeCrystalInputs:
    workdir = Path(workdir)
    return MakeCrystalInputs(
        workdir=workdir,
        prot_pdb=workdir / "prot.pdb",
        pdb_clean=workdir / "pdb_clean.pdb",
    )


def check_inputs(inputs: MakeCrystalInputs) -> None:
    inputs.check_exists(STEP)


def make_crystal(
    inputs: MakeCrystalInputs,
    params: CrystalParams,
    *,
    resume: bool = False,
) -> MakeCrystalResult:
    # local imports keep module import (and the contract tests) light
    from ..crystal import expand_to_unit_cell, gemmi_version, propagate_cell
    from ..gmx import run_tool

    check_inputs(inputs)
    wd = inputs.workdir
    ix = params.ix
    iy = params.iy if params.iy is not None else ix
    iz = params.iz if params.iz is not None else ix

    prot_dry = wd / "prot_dry.pdb"
    prot_dry_cell = wd / "prot_dry_cell.pdb"
    xtal = wd / "xtal.pdb"

    outputs = {"prot_dry": prot_dry, "prot_dry_cell": prot_dry_cell, "xtal": xtal}
    consumed = {"prot_pdb": inputs.prot_pdb, "pdb_clean": inputs.pdb_clean}

    if resume and xtal.exists():
        return MakeCrystalResult(
            step=STEP,
            status=StepStatus.SKIPPED,
            workdir=wd,
            outputs=outputs,
            params=params.model_dump(),
            input_checksums=checksums(consumed),
        )

    # 1. Dry the protein (strip waters/ions) -> prot_dry.pdb
    run_tool(
        ["pdb4amber", "-i", str(inputs.prot_pdb), "-o", str(prot_dry), "--dry"],
        tool="pdb4amber",
        cwd=wd,
        log_path=wd / "pdb4amber_dry.log",
    )

    # 2. Restore the CRYST1 cell from pdb_clean; drop stray ions / any wrong CRYST1.
    _prepend_cryst1(inputs.pdb_clean, prot_dry)

    # 3. Expand the asymmetric unit to the full unit cell, already in spacegroup P1.
    expanded = expand_to_unit_cell(
        prot_dry,
        prot_dry_cell,
        spacegroup=params.spacegroup,
        op_order=params.op_order,
    )

    # 4. Replicate the P1 cell into the requested supercell.
    metrics: dict[str, float | int] = {
        "ix": ix,
        "iy": iy,
        "iz": iz,
        "cell_atoms": expanded.natoms,
        "cell_copies": len(expanded.chain_ids),
    }
    if ix > 0 or iy > 0 or iz > 0:
        propagated = propagate_cell(
            prot_dry_cell,
            xtal,
            ix=ix,
            iy=iy,
            iz=iz,
            numbering=params.numbering,
        )
        metrics["xtal_atoms"] = propagated.natoms
        metrics["xtal_residues"] = propagated.nresidues
    else:
        shutil.copy(prot_dry_cell, xtal)

    return MakeCrystalResult(
        step=STEP,
        status=StepStatus.COMPLETED,
        workdir=wd,
        outputs=outputs,
        params=params.model_dump(),
        input_checksums=checksums(consumed),
        tool_versions={"gemmi": gemmi_version()},
        metrics=metrics,
    )


def _prepend_cryst1(source_pdb: Path, target_pdb: Path) -> None:
    """Copy CRYST1 from source to the top of target, dropping Na+/Cl-/old CRYST1."""
    cryst1 = ""
    with open(source_pdb) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                cryst1 = line
                break

    with open(target_pdb) as fh:
        lines = fh.readlines()
    filtered = [
        line
        for line in lines
        if "Na+" not in line and "Cl-" not in line and not line.startswith("CRYST1")
    ]
    with open(target_pdb, "w") as fh:
        fh.write(cryst1)
        fh.writelines(filtered)
