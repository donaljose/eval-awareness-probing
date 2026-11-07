import json, sys, random
d = json.load(open(sys.argv[1]))
# baseline responses (from per_prompt) - already accessible
random.seed(0)
sample = random.sample(d['per_prompt'], 10)
for r in sample:
 print(f"[{r['framing']:10}] refused={r['refused']}: {r['response'][:200]!r}")
 print()
