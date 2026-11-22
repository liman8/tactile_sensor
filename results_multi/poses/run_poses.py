"""Phase 1.3: stable resting poses of the new objects, with collect_dataset.stable_poses (the rules used for fork_stl:
trimesh candidates; a release 1 mm above the pose; 20 releases tilted 0-2 deg; 500 random drops; natural = the 1 mm
release passes AND >= 19/20 tilted releases pass AND drop frequency >= 1 %; a side-edge pose that is not natural is
"propped", excluded from sampling by default).
    python results_multi/poses/run_poses.py NAME [NAME ...]   -> dataset_multi/poses/<name>/stable_poses.{json,npz,png}
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import collect_dataset as cd  # noqa: E402

if __name__ == "__main__":
    for name in sys.argv[1:]:
        cd.use_object(name, scene="gelsight_multi.xml")
        print(f"=== {name}: scene {os.path.relpath(cd.SCENE_XML, ROOT)}, parts {cd.PARTS}", flush=True)
        cd.stable_poses(os.path.join(ROOT, "dataset_multi", "poses", name), n_drops=500, workers=24)
