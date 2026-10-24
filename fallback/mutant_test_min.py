"""Fallback mutant-separation test with automatic operator-swap mutants.
python fallback/mutant_test_min.py  -> fallback/mutant_scores.json
"""
import ast, json, random, sys
import torch.nn.functional as F
sys.path.insert(0, "fallback")
from build_cache_min import calls, probes, run, embed
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Add: ast.Sub, ast.Sub: ast.Add}

class Mut(ast.NodeTransformer):
    def __init__(self, k):
        self.k, self.i = k, 0
    def _m(self, op):
        if type(op) in SWAP:
            self.i += 1
            if self.i == self.k:
                return SWAP[type(op)]()
        return op
    def visit_Compare(self, n):
        self.generic_visit(n); n.ops = [self._m(o) for o in n.ops]; return n
    def visit_BinOp(self, n):
        self.generic_visit(n); n.op = self._m(n.op); return n

enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
ds = load_dataset("google-research-datasets/mbpp", "full")["train"]
random.seed(0)
beh, txt, other, refs = [], [], [], []
for ex in random.sample(list(ds), 120):
    pr = probes(calls(ex["test_list"]))
    ref = run(ex["code"], ex["test_setup_code"], pr)
    if not ref:
        continue
    zr = embed(enc, ref); refs.append(zr)
    for k in (1, 2, 3):
        try:
            mcode = ast.unparse(Mut(k).visit(ast.parse(ex["code"])))
        except Exception:
            continue
        mt = run(mcode, ex["test_setup_code"], pr)
        if mt and mt != ref:
            beh.append(F.cosine_similarity(zr, embed(enc, mt), dim=0).item())
            ct = enc.encode([ex["code"], mcode], convert_to_tensor=True, normalize_embeddings=True)
            txt.append(F.cosine_similarity(ct[0], ct[1], dim=0).item())
            break
for a, b in zip(refs, refs[1:]):
    other.append(F.cosine_similarity(a, b, dim=0).item())
m = lambda x: round(sum(x) / max(len(x), 1), 3)
res = {"n_mutants": len(beh), "behaviour_ref_vs_mutant": m(beh), "codetext_ref_vs_mutant": m(txt),
       "behaviour_ref_vs_other_task": m(other)}
print(res)
json.dump({**res, "behaviour": beh, "codetext": txt, "other": other}, open("fallback/mutant_scores.json", "w"))
