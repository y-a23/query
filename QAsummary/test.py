import pandas as pd
import numpy as np
import pandas as pd, numpy as np
df = pd.read_parquet('/nfsdata3/yiao/data/PaperSearchQA_summary/data/merge_data/train_merged_by_pmid.parquet')
for _, r in df.iterrows():
    qs = r['questions']
    if isinstance(qs, (list, tuple, np.ndarray)):
        for i, q in enumerate(qs):
            if not isinstance(q, dict):
                print("bad:", r["pmid"], "q_idx", i, "type", type(q), q)
                raise SystemExit