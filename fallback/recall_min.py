"""python fallback/recall_min.py MODEL_DIR CACHE   [WS, env train]"""
import json, sys
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
mdir, cpath = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(mdir)
m = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.bfloat16).cuda().eval()
head = torch.nn.Linear(m.config.hidden_size, 768, bias=False)
head.load_state_dict(torch.load(f"{mdir}/pred_head.pt")); head = head.cuda()
cache = torch.load(cpath); ids = sorted(cache)
Z = F.normalize(torch.stack([cache[i] for i in ids]).float(), dim=1).cuda()
hits = n = 0
for r in map(json.loads, open("datasets/mbpp_train.jsonl")):
    if r["task_id"] not in cache:
        continue
    text = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
    x = tok(text, return_tensors="pt").to("cuda")  # add_special_tokens=True, same as finetune.py
    with torch.no_grad():
        h = m(**x, output_hidden_states=True).hidden_states[-1][0, -1].float()
    hits += ids.index(r["task_id"]) in (Z @ F.normalize(head(h), dim=0)).topk(5).indices.tolist(); n += 1
print("Recall@5", round(hits / n, 4), "n", n)
