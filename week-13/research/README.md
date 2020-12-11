# Research Stage

Hi! This is the first part of the extra lesson. 
Here you'll find some code examples, that will be used to train simple classification model.

## Instruction

### Docker

To run docker and download necessary data files, follow the steps:

```bash
1. cd docker && ./docker.sh {WORK_DIR}
2. ./download_data.sh {DATA_DIR} 
```

If you run experiments without docker, you need to set `DATA_PATH` variable:

```bash
export DATA_PATH={DATA_DIR}
```

### Experiments

In ``src`` folder you can find common used file scripts and experiments configurations. 
Each experiment consist of:
- `config.yaml` - Experiment parameters
- `experiment.py` - Main experiment scripts
- `__init__.py` - Model & Experiment file imports

To run `simple` or `dist` examples, use command(in `src` folder):
```bash
catalyst-dl run --config=experiments/{simple,dist}/config.yaml
```

To find the best hyperparameters by optuna(`optuna` and `multistage`):

```bash
catalyst-dl tune --config=experiments/{optuna,multistage}/config.yaml
```