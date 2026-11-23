# YCB models for the 10-object extension (candidates)

Source: the YCB Object and Model Set (Calli et al., 2015/2017), Google scanner meshes `google_16k`, downloaded
2026-11-23 from `https://ycb-benchmarks.s3.amazonaws.com/data/google/<object>_google_16k.tgz` into
`downloads/ycb/<object>/` (not in git). License: CC BY 4.0 (attribution: the YCB Object and Model Set).

* `inspect_ycb.py` / `inspect_ycb.json`: SHA-256 of each `nontextured.stl` (metres), triangles, watertightness.
* `inspect_ycb2.py`: true size (oriented box), extra components, quasi-static stable poses (trimesh, >= 1 %).
* `spoon_repair_check.py`: 031_spoon is not watertight only because of 2 extra components (4 faces) lying within
  0.0007 mm of the main surface; dropping them gives a watertight mesh (geometry rule: < 0.02 mm, allowed).

| object | size (mm) | triangles | watertight | stable poses >= 1 % (trimesh) | note |
|---|---|---|---|---|---|
| 030_fork | 197.8 x 25.6 x 15.5 | 16,384 | yes (genus 11: scan tunnels) | 6 | |
| 031_spoon | 195.7 x 41.5 x 20.4 | 16,384 | after dropping 4 faces (0.0007 mm) | 3 | |
| 032_knife | 214.9 x 19.5 x 15.3 | 16,384 | yes (genus 30: scan tunnels) | 8 | |
| 033_spatula | 311.0 x 83.6 x 32.9 | 16,384 | yes | 2 | longer than the belief grid (+-100 mm) and most print beds (diagonal only) |
| 043_phillips_screwdriver | 216.2 x 34.1 x 33.5 | 16,384 | yes | 33 | round handle: rolls; no discrete stable poses |
| 044_flat_screwdriver | 216.2 x 34.5 x 33.6 | 16,384 | yes | 35 | round handle: rolls; no discrete stable poses |
