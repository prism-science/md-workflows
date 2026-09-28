"""Crystallographic primitives: symmetry expansion and lattice propagation.

These replace the two external tools ``make_crystal`` used to shell out to — ChimeraX
(``changechains`` / ``unitcell`` / ``combine``) and AmberTools ``PropPDB`` — with gemmi,
so building the crystal needs neither a graphical nor an Amber installation.

Both primitives rewrite the fixed-column ``ATOM``/``HETATM`` text of the input rather
than round-tripping through a :class:`gemmi.Structure`: atom order, residue names and
numbers, and the ATOM/HETATM record types then survive byte-for-byte, which the
atom-order-dependent downstream topologies (tleap, GROMACS) require. For the same
reason the serial/resSeq fields stay plain decimal — gemmi's own ``write_pdb`` switches
to hybrid-36 past 99999 atoms and ``gmx solvate``, the consumer of ``xtal.pdb``, cannot
read that. Supercells too large for those fields are handled by
:func:`propagate_cell`'s ``per-cell`` numbering instead.

All failures raise structured exceptions; nothing here calls ``sys.exit``.
"""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from pathlib import Path

import gemmi
from pydantic import BaseModel

from .config import CrystalNumbering
from .exceptions import CrystalSymmetryError

_COORD_RECORDS = ("ATOM  ", "HETATM")

# Single-character chain ids handed out to symmetry copies and to propagated cells.
_CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits

# Capacity of the fixed-column PDB serial (5 cols) and resSeq (4 cols) fields.
_MAX_SERIAL = 99999
_MAX_RESSEQ = 9999


class ExpandResult(BaseModel):
    """Unit cell written by :func:`expand_to_unit_cell`."""

    cell_pdb: Path
    spacegroup: str  # space group the asymmetric unit was expanded with
    chain_ids: list[str]  # chain id per symmetry copy, in output order
    natoms: int


class PropagateResult(BaseModel):
    """Supercell written by :func:`propagate_cell`."""

    out_pdb: Path
    numbering: CrystalNumbering  # scheme actually used ("auto" already resolved)
    ncells: int
    natoms: int
    nresidues: int


def gemmi_version() -> str:
    """gemmi version string, for a step's ``tool_versions`` metadata."""
    return gemmi.__version__


# --------------------------------------------------------------------------- #
# Fixed-column PDB text helpers
# --------------------------------------------------------------------------- #
def _pad(line: str, width: int = 80) -> str:
    """Right-pad a record so every fixed-column field can be sliced and rewritten."""
    return line.rstrip("\n").ljust(width)


def _is_coord(line: str) -> bool:
    return line.startswith(_COORD_RECORDS)


def _get_xyz(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def _set_xyz(line: str, x: float, y: float, z: float) -> str:
    line = _pad(line)
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def _set_chain(line: str, chain: str) -> str:
    line = _pad(line)
    return line[:21] + chain[0] + line[22:]


def _set_serial(line: str, serial: int) -> str:
    line = _pad(line)
    return f"{line[:6]}{serial % (_MAX_SERIAL + 1):5d}{line[11:]}"


def _set_resseq(line: str, resseq: int) -> str:
    line = _pad(line)
    return f"{line[:22]}{resseq % (_MAX_RESSEQ + 1):4d}{line[26:]}"


def _set_segid(line: str, segid: str) -> str:
    """Write the segID field (columns 73-76)."""
    line = _pad(line)
    return f"{line[:72]}{segid[:4]:<4}{line[76:]}"


def _residue_key(line: str) -> tuple[str, str, str, str]:
    """Chain id, residue name, residue sequence number and insertion code."""
    line = _pad(line)
    return line[21], line[17:20], line[22:26], line[26]


def _count_residues(lines: Sequence[str]) -> int:
    """Number of residues a PDB reader would delimit in ``lines``."""
    count = 0
    previous = None
    for line in lines:
        if _is_coord(line):
            key = _residue_key(line)
            if key != previous:
                count += 1
                previous = key
    return count


def _format_cryst1(cell: gemmi.UnitCell, spacegroup: str = "P 1") -> str:
    return (
        f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
        f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} {spacegroup:<11}"
    )


def _read_pdb(pdb: Path) -> tuple[str, list[str]]:
    """Return the CRYST1 record and the ATOM/HETATM/TER body of a PDB file.

    Headers, CONECT and secondary-structure records are dropped: ChimeraX carried some
    of them through, but nothing downstream in this pipeline reads them and they would
    be stale after expansion.
    """
    cryst1: str | None = None
    body: list[str] = []
    with open(pdb) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                if cryst1 is None:
                    cryst1 = _pad(line)
            elif _is_coord(line) or line.startswith("TER"):
                body.append(_pad(line))
    if cryst1 is None:
        raise CrystalSymmetryError(f"{pdb}: no CRYST1 record, so the unit cell is unknown")
    return cryst1, body


def _write_pdb(pdb: Path, lines: Sequence[str]) -> None:
    pdb.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def _cell_from_cryst1(cryst1: str) -> gemmi.UnitCell:
    return gemmi.UnitCell(
        float(cryst1[6:15]),
        float(cryst1[15:24]),
        float(cryst1[24:33]),
        float(cryst1[33:40]),
        float(cryst1[40:47]),
        float(cryst1[47:54]),
    )


def _spacegroup_from_cryst1(cryst1: str, override: str | None = None) -> gemmi.SpaceGroup:
    name = override if override else cryst1[55:66].strip()
    if not name:
        raise CrystalSymmetryError("CRYST1 carries no space group; set crystal.spacegroup")
    spacegroup = gemmi.find_spacegroup_by_name(name)
    if spacegroup is None:
        raise CrystalSymmetryError(f"unknown space group {name!r}")
    return spacegroup


def _ordered_operations(spacegroup: gemmi.SpaceGroup, op_order: Sequence[int] | None) -> list:
    """Symmetry operations of ``spacegroup``, optionally reordered.

    gemmi lists the operations in Hall-symbol order. For P 21 21 21 that swaps the third
    and fourth relative to the International Tables order ChimeraX used, which swaps the
    contents of chains C and D; ``op_order=[1, 2, 4, 3]`` reproduces the ChimeraX
    A/B/C/D labelling exactly.
    """
    ops = list(spacegroup.operations())
    if op_order is None:
        return ops
    if sorted(op_order) != list(range(1, len(ops) + 1)):
        raise CrystalSymmetryError(
            f"op_order must be a permutation of 1..{len(ops)} for {spacegroup.xhm()}, "
            f"got {list(op_order)}"
        )
    return [ops[i - 1] for i in op_order]


def _lattice_vectors(cell: gemmi.UnitCell) -> list[gemmi.Position]:
    """Cartesian a, b, c vectors of the cell.

    Orthogonalization is linear, so the fractional basis vectors map straight onto them.
    """
    bases = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return [cell.orthogonalize(gemmi.Fractional(*basis)) for basis in bases]


# --------------------------------------------------------------------------- #
# Asymmetric unit -> unit cell (replaces ChimeraX)
# --------------------------------------------------------------------------- #
def expand_to_unit_cell(
    asu_pdb: str | Path,
    cell_pdb: str | Path,
    *,
    spacegroup: str | None = None,
    chain_ids: str | None = None,
    op_order: Sequence[int] | None = None,
) -> ExpandResult:
    """Expand one asymmetric unit into the full unit cell, written in space group P 1.

    Stands in for ChimeraX's ``changechains #1 A; unitcell #1; combine #2``: every
    symmetry operation of the CRYST1 space group is applied to the input coordinates and
    each copy is then displaced by the integer lattice vector that brings its
    (unweighted) atom centroid into ``[0, 1)^3`` — the packing rule ``unitcell`` uses.
    Copies are labelled ``A``, ``B``, ``C``, ... and the output CRYST1 already carries
    ``P 1``, so no separate space-group rewrite is needed.
    """
    asu_pdb, cell_pdb = Path(asu_pdb), Path(cell_pdb)
    cryst1, body = _read_pdb(asu_pdb)
    cell = _cell_from_cryst1(cryst1)
    group = _spacegroup_from_cryst1(cryst1, spacegroup)
    ops = _ordered_operations(group, op_order)

    ids = chain_ids if chain_ids else _CHAIN_IDS
    if len(ops) > len(ids):
        raise CrystalSymmetryError(
            f"{group.xhm()} needs {len(ops)} chain ids for its symmetry copies but only "
            f"{len(ids)} are available"
        )

    fractional = [
        cell.fractionalize(gemmi.Position(*_get_xyz(line))).tolist()
        for line in body
        if _is_coord(line)
    ]

    out = [_format_cryst1(cell)]
    used_ids: list[str] = []
    serial = 0
    for icopy, op in enumerate(ops):
        chain = ids[icopy]
        used_ids.append(chain)
        moved = [op.apply_to_xyz(frac) for frac in fractional]
        centroid = [sum(values) / len(moved) for values in zip(*moved, strict=True)]
        shift = [-math.floor(value) for value in centroid]
        packed = [
            cell.orthogonalize(
                gemmi.Fractional(frac[0] + shift[0], frac[1] + shift[1], frac[2] + shift[2])
            )
            for frac in moved
        ]
        icoord = 0
        for line in body:
            if _is_coord(line):
                pos = packed[icoord]
                icoord += 1
                serial += 1
                line = _set_xyz(line, pos.x, pos.y, pos.z)
            out.append(_set_serial(_set_chain(line, chain), serial))
    out.append("END")

    _write_pdb(cell_pdb, out)
    return ExpandResult(
        cell_pdb=cell_pdb,
        spacegroup=group.xhm(),
        chain_ids=used_ids,
        natoms=serial,
    )


# --------------------------------------------------------------------------- #
# Unit cell -> supercell (replaces AmberTools PropPDB)
# --------------------------------------------------------------------------- #
def propagate_cell(
    cell_pdb: str | Path,
    out_pdb: str | Path,
    *,
    ix: int = 1,
    iy: int = 1,
    iz: int = 1,
    numbering: CrystalNumbering = "auto",
) -> PropagateResult:
    """Replicate a P 1 unit cell ``ix`` x ``iy`` x ``iz`` times and scale CRYST1.

    Stands in for AmberTools ``PropPDB``, including its replication order (z fastest,
    then y, then x), so the output is layout-compatible with it.

    ``numbering`` selects how atoms and residues are counted:

    - ``continuous`` — sequential across the whole supercell, which is what PropPDB
      writes. The serial field holds 99999 atoms and resSeq 9999 residues; past that
      the counters wrap.
    - ``per-cell`` — restart both counters in every unit cell, give each (cell, input
      chain) pair its own chain id and record the cell index in the segID field, so
      every field stays inside its column width however large the supercell grows.
    - ``auto`` (default) — ``continuous`` while it fits those fields, ``per-cell``
      beyond that; identical to ``continuous`` for every supercell it can represent.
    """
    cell_pdb, out_pdb = Path(cell_pdb), Path(out_pdb)
    if min(ix, iy, iz) < 1:
        raise CrystalSymmetryError(f"propagation counts must be >= 1, got {ix}, {iy}, {iz}")

    cryst1, body = _read_pdb(cell_pdb)
    cell = _cell_from_cryst1(cryst1)
    avec, bvec, cvec = _lattice_vectors(cell)
    supercell = gemmi.UnitCell(
        cell.a * ix, cell.b * iy, cell.c * iz, cell.alpha, cell.beta, cell.gamma
    )

    ncells = ix * iy * iz
    natoms_in = sum(1 for line in body if _is_coord(line))
    nres_in = _count_residues(body)
    natoms = natoms_in * ncells
    nresidues = nres_in * ncells

    mode = numbering
    if mode == "auto":
        fits = natoms <= _MAX_SERIAL and nresidues <= _MAX_RESSEQ
        mode = "continuous" if fits else "per-cell"

    in_chains: list[str] = []
    for line in body:
        if _is_coord(line) and line[21] not in in_chains:
            in_chains.append(line[21])
    if mode == "per-cell" and (natoms_in > _MAX_SERIAL or nres_in > _MAX_RESSEQ):
        raise CrystalSymmetryError(
            f"{cell_pdb}: one unit cell ({natoms_in} atoms, {nres_in} residues) already "
            "overflows the fixed-column PDB serial/resSeq fields"
        )

    out = [_format_cryst1(supercell)]
    serial = 0
    resseq = 0
    previous_key = None
    chain_map: dict[str, str] = {chain: chain for chain in in_chains}
    segid = ""
    icell = -1
    # z fastest, then y, then x — the order PropPDB writes its copies in.
    for cx in range(ix):
        for cy in range(iy):
            for cz in range(iz):
                icell += 1
                dx = cx * avec.x + cy * bvec.x + cz * cvec.x
                dy = cx * avec.y + cy * bvec.y + cz * cvec.y
                dz = cx * avec.z + cy * bvec.z + cz * cvec.z

                if mode == "per-cell":
                    serial = 0
                    resseq = 0
                    previous_key = None
                    base = icell * len(in_chains)
                    chain_map = {
                        chain: _CHAIN_IDS[(base + offset) % len(_CHAIN_IDS)]
                        for offset, chain in enumerate(in_chains)
                    }
                    segid = f"{icell:04d}"

                for line in body:
                    if mode == "per-cell":
                        line = _set_segid(_set_chain(line, chain_map[line[21]]), segid)
                    if _is_coord(line):
                        key = (cx, cy, cz, *_residue_key(line))
                        if key != previous_key:
                            resseq += 1
                            previous_key = key
                        serial += 1
                        x, y, z = _get_xyz(line)
                        line = _set_xyz(line, x + dx, y + dy, z + dz)
                    out.append(_set_resseq(_set_serial(line, serial), resseq))
    out.append("END")

    # A wrapped resSeq can merge two neighbouring residues into one as far as any PDB
    # reader is concerned, which silently corrupts the topology; refuse to write that.
    delimited = _count_residues(out)
    if delimited != nresidues:
        hint = "; use crystal.numbering='per-cell'" if mode == "continuous" else ""
        raise CrystalSymmetryError(
            f"{out_pdb}: {nresidues} residues were propagated but a PDB reader would "
            f"delimit {delimited} — adjacent residues share chain, name and number{hint}"
        )

    _write_pdb(out_pdb, out)
    return PropagateResult(
        out_pdb=out_pdb,
        numbering=mode,
        ncells=ncells,
        natoms=natoms,
        nresidues=nresidues,
    )
