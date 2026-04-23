# Environment Repro (Conda)

Use `environment.conda.yml`.

## Create

```bash
conda env create -f environment.conda.yml
conda activate get
```

## Update

```bash
conda env update -n get -f environment.conda.yml --prune
```

## Notes

- DGL is installed from `dglteam/label/th24_cu124` (Torch 2.4 + CUDA 12.4 line).
- Equivalent direct command:

```bash
conda install -c dglteam/label/th24_cu124 dgl
```
