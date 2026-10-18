# Cognitive Maps in Large Language Models

Code and analysis for **Cognitive Maps in Large Language Models**. 

This repository contains two Python packages:

- **`llm-neuroscience-tools`** provides model wrappers, prompt generation,
  activation and key-value cache interventions, and an interface for mosaic
  experiments.
- **`cognitive-maps-in-llms`** contains the paper's experiments, analysis
  utilities, and figure generation notebook/scripts.

Figures can be reproduced from the supplied results without rerunning the 
experiments. Running new experiments requires separately downloaded model
weights and suitable GPU hardware.

## Installation

Clone this repository and enter its root directory:

```bash
git clone https://github.com/albertqin8221/cognitive-maps-in-llms.git
cd cognitive-maps-in-llms
```

Create and activate a Conda environment:

```bash
conda create -n cognitive-maps python=3.13 pip
conda activate cognitive-maps
```

Install both packages:

```bash
python -m pip install ./llm-neuroscience-tools
python -m pip install ./cognitive-maps-in-llms
```

The Python import names are `llm_neuroscience_tools` and
`cognitive_maps_in_llms`.

## Data download

Download `cognitive-maps-results.tar.gz` from this repository's GitHub Releases
page and save it in the root of this repository. From the repository root, 
extract the archive into the paper package:

```bash
tar -xzf cognitive-maps-results.tar.gz -C cognitive-maps-in-llms
```

After extraction, the repository should have this layout:

```text
.
├── README.md
├── llm-neuroscience-tools/
└── cognitive-maps-in-llms/
    ├── figures/
    │   └── notebooks/
    ├── results/
    │   └── final_run_2026-09-30/
    │       └── <model-preset>/
    │           └── <prompt-preset>/
    │               └── <task-name>/
    └── ...
```

The notebooks use this location by default. The data are organized by model, prompt preset, and task:

| Output | Contents |
| --- | --- |
| `exp_*/results.ndjson` | Results of mosaic experiments. |
| `iterative_erasure/` | Pruning trajectories describing the erasure procedure. |
| `pca_inputs/` | Selected activation recordings used by the supplementary PCA figure. |

## Reproducing figures

After downloading the results, use the notebooks in
[`figures/notebooks/`](cognitive-maps-in-llms/figures/notebooks/)
to reproduce the paper’s figures. See the
[figure-generation instructions](cognitive-maps-in-llms/README.md#quickstart-generating-figures)
for additional documentation.

## Running new experiments

To run new experiments, follow the
[model setup](llm-neuroscience-tools/README.md#downloading-models)
and [experiment orchestration](cognitive-maps-in-llms/README.md#quickstart-running-gpu-tasks)
instructions. The repository includes SLURM planning tools configured for 
Princeton's `Della` cluster.

## Additional documentation

For tools and APIs for experiments of the general form described in the paper, see the 
[`llm-neuroscience-tools` README](llm-neuroscience-tools/README.md).

For paper-specific experiments and analyses, start with the
[`cognitive-maps-in-llms` README](cognitive-maps-in-llms/README.md).

## License

The code and documentation in both packages are distributed under the
[MIT License](LICENSE). The accompanying results released through GitHub Releases
are licensed under [CC BY 4.0](LICENSE-DATA.md).

Third-party software and model weights remain subject to their respective licenses.
