import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "runs/run_qwen1.5_1.8b_v5_4/results.json"
d = json.load(open(path))
print(f"model: {d['config']['model_name']}")
print(f"steering: {d.get('steering_method', 'ablation')}")
print(f"verdict: {d['suppression']['overall_verdict']}")
print()
print("alpha  AUC     G       d_auc    d_gap    verdict         oversteered")
for p in d['suppression']['sweep']:
    print(f"{p['alpha']:<5}  {p['auc']:.3f}   {p['gap']:.3f}   {p['d_auc']:+.3f}   {p['d_gap']:+.3f}   {p['verdict']:<14}  {p['oversteered']}")
