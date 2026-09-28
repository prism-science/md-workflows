# md-workflows

A Python CLI and Docker image for running the full molecular-dynamics workflow pipeline
(GROMACS + AmberTools + DIALS/cctbx + lunus).

## Docker image layout

The image is built in stages so the architecture-specific GROMACS compile is isolated from
the portable scientific stack — forks targeting other hardware can swap the GROMACS stage
without rebuilding the base:

| Stage | Dockerfile | Contents |
|-------|-----------|----------|
| base    | `Dockerfile.base`    | CUDA 12.6 devel toolchain, micromamba/conda `lunus` env, lunus, ChimeraX. Architecture-neutral. |
| gromacs | `Dockerfile.gromacs` | GROMACS (CUDA, tuned for H100 / AVX-512 by default) + the `md_workflows` package. The consumable image. |

(An Astera-specific `Dockerfile.actl` overlay adds workspace conventions on top of `gromacs`. It
is tracked here so it stays under CI lint coverage, but is built and published only from the
`astera` deployment branch.)

## 1) Build the images

From the project root:

```bash
# 1. Base — architecture-neutral foundation
docker build -f Dockerfile.base -t md-base:local .

# 2. GROMACS + md-workflows (the runnable image).
#    Override the GMX_* build args for non-H100 / non-AVX-512 hardware, e.g.
#    --build-arg GMX_CUDA_TARGET_SM=80 --build-arg GMX_SIMD=AVX2_256
docker build -f Dockerfile.gromacs --build-arg BASE_IMAGE=md-base:local -t md-gromacs:local .
```

CI builds these stages and pushes versioned tags (derived from the `version` in
`pyproject.toml`) to the Astera Harbor registry; see `.github/workflows/build-images.yml` on
the `astera` branch.

## 2) Start a container

Run interactively, mounting the project directory so inputs/outputs are available on the host:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  --name md_container \
  -e HOME=/workspace \
  -v "$(pwd):/workspace" \
  -w /workspace \
  md-gromacs:local \
  bash
```

This registers the single `md-workflows` CLI entry point from `pyproject.toml`.

## 3) Run the workflow

The CLI is one Typer app with a subcommand per step plus `run-pipeline`. Global options
`--workdir/-w` (run directory), `--config/-c` (YAML/TOML), and `--resume/--force` come
before the subcommand.

Run the full prep pipeline (`param_prot → make_crystal → make_waterbox → solvate →
minimize → equilibrate → resolvate`, ending with the adaptive pressure loop):

```bash
md-workflows --workdir . run-pipeline --pdb-id 4LZT --ix 5
```

Run a single step (each resolves its inputs from `--workdir` and fails loudly if any are
missing):

```bash
md-workflows -w . minimize
md-workflows -w . resolvate --target-bar 1.0
```

Per-system settings live in a config file; flags override individual values. The
dominant cross-machine knob is the per-invocation GROMACS `mdrun` profile
(`run_profiles`, e.g. GPU offload + thread counts):

```yaml
# config.yaml
pdb_id: 4LZT
crystal: { ix: 5 }
waterbox: { nc_scale: 5, conc: 60.0 }
solvate: { ionic_strength: 0.1 }
run_profiles:
  equil: { ntomp: 16, nb: gpu, pme: gpu, bonded: gpu, tunepme: false }
  min:   { ntomp: 16, nb: gpu, pme: cpu, bonded: cpu, tunepme: false }
```

`make_crystal` expands the asymmetric unit to the unit cell and propagates the lattice
with [gemmi](https://gemmi.readthedocs.io) (`md_workflows/core/crystal.py`) instead of
ChimeraX and AmberTools `PropPDB`, so neither is needed at run time. `crystal.spacegroup`
overrides the CRYST1 space group, `crystal.op_order` reorders the symmetry operations
(e.g. `[1, 2, 4, 3]` reproduces the chain labelling of the old ChimeraX route), and
`crystal.numbering` picks the supercell's atom/residue numbering — `continuous`
(PropPDB-compatible), `per-cell` (counters restart in every cell, for supercells past
the 99999-atom / 9999-residue PDB fields) or `auto`.

```bash
md-workflows -w run_dir -c config.yaml run-pipeline
md-workflows -w run_dir -c config.yaml --resume run-pipeline   # skip completed steps
```

`--resume` skips a step whose durable outputs already exist. Exit codes: `0` ok,
`1` domain error, `2` missing input, `3` external tool failed.

### Python / SDK

The same logic is importable:

```python
from md_workflows import run_standard_md
result = run_standard_md("4LZT", "run_dir", crystal={"ix": 5}, resume=True)
```

To see all commands and flags:

```bash
md-workflows --help
md-workflows run-pipeline --help
```

## Development

Lint and format with [ruff](https://docs.astral.sh/ruff/) (config in `pyproject.toml`):

```bash
pip install '.[dev]'
ruff check md_workflows
ruff format --check md_workflows
```
