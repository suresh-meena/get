import statistics as st
from experiments.common import load_tu_dataset

tu8 = [
    'PROTEINS','NCI1','NCI109','DD','ENZYMES','MUTAG','MUTAGENICITY','FRANKENSTEIN'
]
print('dataset,graphs,feat_dim,num_classes,mean_nodes,p95_nodes,max_nodes,mean_edges,p95_edges,max_edges')
for name in tu8:
    ds = load_tu_dataset(name, limit=None)
    n_graphs = len(ds)
    nodes = [int(g['x'].size(0)) for g in ds]
    edges = [len(g['edges']) for g in ds]
    labels = sorted({int(g['y'].view(-1)[0].item()) if hasattr(g['y'], 'view') else int(g['y']) for g in ds})
    feat_dim = int(ds[0]['x'].size(1)) if ds else 0
    def p95(xs):
        if not xs:
            return 0
        k = int(round(0.95*(len(xs)-1)))
        return sorted(xs)[k]
    print(f"{name},{n_graphs},{feat_dim},{len(labels)},{st.fmean(nodes):.2f},{p95(nodes)},{max(nodes)},{st.fmean(edges):.2f},{p95(edges)},{max(edges)}")
