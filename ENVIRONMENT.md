# Environment Repro

Use `requirements.txt`.

## Create

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Notes

- If you want a different CUDA or CPU-only build, swap the wheel index lines in `requirements.txt` to the matching ones.
