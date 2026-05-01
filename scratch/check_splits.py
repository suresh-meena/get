import torch
from ogb.graphproppred import PygGraphPropPredDataset

def main():
    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root="data/OGB")
    split = ds.get_idx_split()
    print(f"Train: {len(split['train'])}, Valid: {len(split['valid'])}, Test: {len(split['test'])}")

if __name__ == "__main__":
    main()
