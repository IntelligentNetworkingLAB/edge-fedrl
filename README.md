# Edge FedRL — Federated Reinforcement Learning for Drifting Edge Nodes

Several edge nodes learn the same control task, each with its own hardware characteristics, and one node's environment changes partway through training. The question is what the nodes should exchange so that the drifting node adapts quickly without moving raw experience off the device, and at what communication cost.

Each node runs its own SAC agent (PyTorch) on a MuJoCo task (`Hopper-v5` by default) with a different actuator strength. At the midpoint, one node's actuator weakens (`EDGE-03`: 1.00 → 0.65). The compared methods:

| Method | What is shared | Per node per round |
|---|---|---|
| `local`, `local_per` | nothing (uniform / prioritized replay) | 0 |
| `fedavg` | full actor and critic parameters, averaged at the server | the whole model, up and down |
| `proto_per` | anonymized prototypes of near-failure states (K-means over an 8-dim shared random projection) | at most 16 × 11 floats (< 1 KB) |
| `private_proto_per` | the same prototypes with Gaussian noise on the centroids | at most 16 × 11 floats (< 1 KB) |

Received prototypes never become training data. They only raise the replay priority of local transitions that look like another node's failures: `P(τ) = (|δ|+ε)^α · (1 + λ·q(τ))`. The critic is always trained on local transitions.

Content-free controls (`noisy_per`, `proto_random`, `proto_shuffled`) and local replay baselines from the literature (`per_ere`, `per_tgr`) are included so that any gain can be attributed correctly.

## Getting started

```bash
pip install -r requirements.txt
python run.py                       # quick preset, console report (a few minutes per method on CPU)
python run.py --all                 # also writes out/report.md and out/result.json
python run.py --preset full         # paper-scale setting (several hours per method and seed)
python run.py --methods local_per proto_per --seeds 0 1 2
```

The nodes, the drift and the presets are defined in `config/experiment.json`.

## Report

For each method the report shows:

- **Node-level performance**: final evaluation return of every edge node, with the median and IQR across nodes.
- **Adaptation of the drifting node**: return before the drift, the area under the evaluation-return curve after it (post-drift AUC), the change against the baseline (`local_per`), how many seeds recovered to 80% of the pre-drift return, and the number of failed episodes right after the drift.
- **Communication**: total bytes exchanged between nodes and the server.

Each run also leaves `eval.csv`, `episodes.csv`, `metrics.json` and, for prototype methods, `prototypes.npz` under `out/runs/<preset>/`.

## Privacy check

The prototypes saved by a run can be attacked directly: an MLP tries to tell which node produced a prototype and to regress its actuator strength, trained on early rounds and tested on later ones.

```bash
python scripts/attack_eval.py out/runs/full/proto_per_seed0/prototypes.npz
```

## Findings so far

The full study (285 runs, Hopper and Walker2d, 5 seeds, paired branches from shared checkpoints) is written up in Korean in [`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/INSIGHTS.md`](docs/INSIGHTS.md). In short:

- Prototype sharing improved the drifting node's post-drift AUC by about 13%, but a content-free control with the same structure gave about 11%. Most of the gain came from perturbing replay priorities, not from transferring experience.
- What drove adaptation was how much sampling mass the replay gave to post-drift data. Forcing 75% of each batch from post-drift data gave +35% without deleting anything from the buffer.
- Existing recency samplers (ERE) matched an oracle that knows the drift time, so they are the baseline any sharing scheme has to beat.

## Project layout

```
run.py              experiment runner and report
fedrl/              SAC, FedAvg, prioritized replay, shared projection, prototypes, privacy attacks
fedrl/train.py      multi-node training loop and metrics
fedrl/report.py     node-level report
config/             nodes, drift and presets
scripts/            single runs, batch runs, and the study scripts behind docs/
docs/               study results and notes (Korean)
tests/              unittest suite
```

`scripts/run.py`, `run_all.py`, `aggregate.py` and `attack_eval.py` work on their own. The study scripts (`gate1*`, `pivot*`, `ablation*`, `stage*`, `walker.py`) branch from shared checkpoints and write into `gate1/` and `gate1_walker/`, which are not tracked.

## Tests

```bash
python -m unittest discover -s tests
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)),(No.2019-0-01287, Evolvable Deep Learning Model Generation Platform for Edge Computing)

## License

Apache-2.0
