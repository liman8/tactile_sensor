"""Apply the rule for the search's CAD temperature (RUN_LOG): validation searches with tau in {0.027, 0.1, 0.3, 1.0} and
without the CAD check (pose_calib_search/); pick the highest success at ADD < 5 mm (right object, refined pose); within
2 points, the fewer confidently wrong stops (stopped with ADD-S > 5 mm or the wrong object). Writes tau_search (or
"off") into results_multi/pose_calib.json and the table into pose_calib_search/choice.json."""
import glob
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(R, "multi_all", "pose_calib_search")
rows = {}
for tag in ["tau0.027", "tau0.1", "tau0.3", "tau1.0", "noverify"]:
    E = sum((json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, f"episodes_{tag}_*.json")))), [])
    if not E:
        continue
    r = [e["refined"] for e in E]
    rc = [e["refined_cad_only"] for e in E]
    f = [e["final"] for e in E]
    add = lambda rr: np.array([x["add_mm"] if x["add_mm"] is not None else np.inf for x in rr])
    wrong_stop = np.mean([e["stopped"] and (e["refined"]["adds_mm"] > 5 or not e["refined"]["identity_ok"]) for e in E])
    rows[tag] = dict(n=len(E), success_add5=float(np.mean(add(r) < 5)), success_add2=float(np.mean(add(r) < 2)),
                     success_add5_cad_only_refine=float(np.mean(add(rc) < 5)), success_add5_binned=float(np.mean(add(f) < 5)),
                     identity=float(np.mean([x["identity_ok"] for x in r])), presses=float(np.mean([e["presses"] for e in E])),
                     stopped=float(np.mean([e["stopped"] for e in E])), wrong_confident_stops=float(wrong_stop),
                     centre_in90=float(np.mean([x["centre_in90"] for x in f])), yaw_in90=float(np.mean([x["yaw_in90"] for x in f])),
                     add_median=float(np.median(add(r)[np.isfinite(add(r))])))
best = max(rows, key=lambda t: rows[t]["success_add5"])
near = [t for t in rows if rows[best]["success_add5"] - rows[t]["success_add5"] <= 0.02]
choice = min(near, key=lambda t: (rows[t]["wrong_confident_stops"], -rows[t]["success_add5"]))
for t, v in rows.items():
    print(f"{t:9s} " + " ".join(f"{k} {v[k]:.3f}" if isinstance(v[k], float) else f"{k} {v[k]}" for k in v) + ("   <- chosen" if t == choice else ""))
json.dump(dict(rows=rows, best_success=best, within_2_points=near, choice=choice), open(os.path.join(D, "choice.json"), "w"), indent=1)
cal = json.load(open(os.path.join(R, "pose_calib.json")))
cal["tau_search"] = None if choice == "noverify" else float(choice[3:])
cal["tau_search_rule"] = "pose_calib_search/choice.json (results_multi/choose_tau.py)"
json.dump(cal, open(os.path.join(R, "pose_calib.json"), "w"), indent=1)
print("chosen:", choice)
