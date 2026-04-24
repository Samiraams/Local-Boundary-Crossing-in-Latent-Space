# Local-Boundary-Crossing-in-Latent-Space

This repository contains the experimental pipeline for local boundary crossing in latent space, along with the instance selection script used to generate the CIFAR-10 instance CSV files.

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

Notes:

- `torch`, `diffusers`, `transformers`, and `datasets` are required.
- The first execution may download models and datasets from Hugging Face.
- If the models are already cached locally, the project can reuse that cache.

## Project Structure

- `run_experiment.py`: main experiment entrypoint.
- `config.yaml`: experiment configuration file.
- `src/`: shared pipeline code and algorithm implementations.
- `scripts/instance_selection.py`: script that generates the instance CSV files.
- `instances/`: CSV files used by the experiments.

The instances used in the experiments are already available in:

- `instances/cifar10_selected_instances_representative_1.csv`

So you do not need to run the instance selection script before running the main experiment unless you want to regenerate the CSV files.

## Main Configuration

The main experiment is controlled by `config.yaml`.

Important parameters in `common`:

- `algorithm`: algorithm to run. Valid values: `genetic`, `cmaes`, `hill`, `random_search`.
- `use_dataset_index_batch`: if `true`, the experiment reads dataset indices from the CSV in `instances_csv_path`. If `false`, it runs on a single local image from `input_image_path`.
- `instances_csv_path`: CSV file with the selected instances. This path is project-relative by default.
- `input_image_path`: local image path used only when `use_dataset_index_batch: false`.
- `runs_per_instance`: number of runs for each instance.
- `num_generations`: number of generations or iterations used by the search process.
- `population_size`: population size or batch size per iteration, depending on the algorithm.
- `batch_eval_size`: batch size used when evaluating candidates with the classifier.
- `save_best_every`: frequency for saving `best_gen_<n>.png`.
- `output_base`: base output directory. If left empty, a default output directory is selected automatically according to the algorithm.
- `instance_limit`: optional limit on how many instances from the CSV are used.
- `target_class`: if `-1`, the predicted class is used automatically.

Algorithm-specific parameters:

- `genetic.elitism`
- `genetic.prob_mutation`
- `genetic.prob_crossover`
- `hill.sigma_up_factor`
- `hill.sigma_down_factor`
- `hill.classifier_eval_budget_in_loop`
- `random_search.classifier_eval_budget_in_loop`

If `classifier_eval_budget_in_loop` is left empty, the code automatically uses:

```text
num_generations * population_size
```

## Running the Main Experiment

From the project root:

```bash
python3 run_experiment.py
```

To use another config file:

```bash
python3 run_experiment.py --config path/to/other_config.yaml
```

### Batch Mode

Default batch mode uses the selected instances CSV:

```yaml
common:
  use_dataset_index_batch: true
  instances_csv_path: instances/cifar10_selected_instances_representative_1.csv
```

### Single-Image Mode

To run with a single local image:

```yaml
common:
  use_dataset_index_batch: false
  input_image_path: path/to/image.png
```

If `input_image_path` is relative, it is resolved relative to the folder containing `config.yaml`.

## Running the Instance Selection Script

The script can be run from the project root:

```bash
python3 scripts/instance_selection.py
```

It supports optional arguments:

```bash
python3 scripts/instance_selection.py \
  --model-name nateraw/vit-base-patch16-224-cifar10 \
  --dataset-name uoft-cs/cifar10 \
  --split test \
  --seed 42 \
  --batch-size 64 \
  --output-dir instances
```

Generated files are saved to `instances/` by default.
