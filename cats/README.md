# CATS: California Test System

`CaliforniaTestSystem.m` is the MATPOWER case from
[WISPO-POP/CATS-CaliforniaTestSystem](https://github.com/WISPO-POP/CATS-CaliforniaTestSystem)
(`MATPOWER/CaliforniaTestSystem.m` at commit `8a6f7fa`), copied unchanged under its
BSD 3-Clause license (`LICENSE`). If you use it, cite:

```bibtex
@article{CATS2024,
  author={Taylor, Sofia and Rangarajan, Aditya and Rhodes, Noah and Snodgrass, Jonathan and Lesieutre, Bernard C. and Roald, Line A.},
  journal={IEEE Transactions on Energy Markets, Policy and Regulation},
  title={California Test System (CATS): A Geographically Accurate Test System Based on the California Grid},
  year={2024},
  volume={2},
  number={1},
  pages={107-118},
  doi={10.1109/TEMPR.2023.3338568}
}
```

The case has 8,870 buses (8,085 PQ, 784 PV, 1 slack), 10,823 branches, 3,892 generators
and 2,472 loads, all in service, with contiguous bus numbers. **1,412 PQ buses have
in-service generators**: PFΔ's graph builder only uses `[pd, qd]` as PQ-bus input, so
their generation must be folded into the input (e.g., `pd - pg`, `qd - qg`) before
training CANOS on this data.

## Generating AC OPF samples

`sample_acopf.jl` writes samples in PFΔ's raw format; see the header of the script for
the sampling procedure and all options.

```bash
julia --project=cats -e 'using Pkg; Pkg.instantiate()'
julia --project=cats cats/sample_acopf.jl --num-samples 1000 --workers 12
```

Samples go to `data/pfdelta_data/cats/n/raw/` by default. Each sample index has its own
seed, so cluster jobs can split the work with `--start`, and re-running skips finished
samples.

Measured on an M4 Max with MUMPS (4-sample test, `pfdelta` formulation): ~9 s per
solve (~10–14 s per sample including I/O), all accepted, **~8.3 MB per sample** on disk.
At PFΔ's 56,000 samples that is about 200 core-hours and ~465 GB.

Note that 2,635 generators have zero cost. At moderate load the zero-cost generators
can cover demand, so the OPF optimum is not unique and their dispatch is whatever
Ipopt converges to (near the warm start). The power flow labels are still exact for
the stored setpoints, but check the spread of generator setpoints in a pilot run.
