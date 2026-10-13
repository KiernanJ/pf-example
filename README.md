This fork of the [pfdelta](https://github.com/MOSSLab-MIT/pfdelta) repository
provides a bridge for embedding a trained CANOS-PF graph neural network in
JuMP/PowerModels optimization models in Julia, via
[MathOptAI.jl](https://github.com/lanl-ansi/MathOptAI.jl). The bridge was
originally developed for the paper "Generating adversarial inputs for a graph
neural network model of AC power flow" (PowerUp 2026):
```bibtex
@misc{parker2026adversarial,
      title={Generating adversarial inputs for a graph neural network model of {AC} power flow}, 
      author={Robert Parker},
      year={2026},
      eprint={2602.17975},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.17975}, 
}
```
See `README-main.md` for the upstream PFΔ dataset and model documentation.

## Workflow

1. `python download.py` &mdash; Download the Case-14 Task 1.1 data to `data/pfdelta_data/`.
2. Train CANOS-PF (`python main.py --config <config>`) or use the committed weights in
   `runs/canos_task_1_1/`.
3. `python test_canos_trained.py` &mdash; Load the trained weights into the `VectorCanos`
   wrapper and save it to `vectorcanos-trained.pt`, which the Julia code loads.
4. `julia --project=. moai-canos.jl` &mdash; Example: embed the trained CANOS model in a
   PowerModels ACPF model at a training point and compare CANOS outputs to ACPF
   outputs and data labels. Use this as a template for your own formulations.

Run all scripts from the repository root, with a Python environment containing
`torch` and `torch_geometric` active (the Julia scripts reuse it via PythonCall).

## Files

- `vectorcanos.py` &mdash; The `VectorCanos` class, which wraps the structured CANOS-PF
  model as a function of flat input/output vectors. When run as a script, saves an
  untrained wrapper to `vector-canos.pt`.
- `moai-canos-model.jl` &mdash; Julia helpers:
  - `get_inputs(pm)` / `get_outputs(pm)` / `get_input_names(pm)` &mdash; PowerModels
    variables/expressions (and bounds) ordered like the CANOS input/output vectors.
  - `load_pfd_into_pm!(pm_data, point)` &mdash; Overwrite PGLib case data with the
    constants in a PFDelta sample.
  - `load_canos(path)` and `add_canos_predictor!(pm, predictor)` &mdash; Load a saved
    wrapper and embed it in a PowerModels JuMP model.
  - `strip_inequalities!(pm)` &mdash; Remove bounds and inequalities, leaving the ACPF
    equations.
  - `solve_powerflow(point)` &mdash; Solve ACPF for a PFDelta sample.
- `moai-canos-mwe.jl` &mdash; Minimal example embedding the untrained `vector-canos.pt`
  in a toy JuMP model.
- `test_vectorcanos.py` &mdash; Checks that the vectorized wrapper reproduces the
  structured model's outputs.
- `validate_canos.py` &mdash; Evaluates loss of the trained CANOS-PF model on train and
  test data.
- `validate-labels.jl` &mdash; Solves ACPF for every Case-14 Task 1.1 training point and
  reports points whose labels differ from the solution by more than `1e-5`.
- `test-data.jl` &mdash; Asserts that constant parameters are consistent between the
  PFDelta dataset and the PGLib Case-14 file.
- `results.jl` &mdash; `results_path(fname)` returns a path under a dated, branch-specific
  `results/<YYYYMMDD>-<branch>/` directory.
