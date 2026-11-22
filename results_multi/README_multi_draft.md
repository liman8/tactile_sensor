## Multi-object: which object is it, and where is its centre? (`results_multi/`)

The fork-only pipeline above (final model `mini_robust` s0) is unchanged and stays reproducible. This part adds three
objects and asks two questions from touch alone: which object is under the pad, and where its centre of mass is, with
an active multi-press search. Every step, decision and number has a line in `results_multi/RUN_LOG.md`.

### 1. Objects and the registry (`objects.json`, `objects.py`)

| object | file | class | mass | centre of mass (STL x, y) | symmetric (outline vs mirror) | face up / down | parts (x from, mm) | CoACD |
|---|---|---|---|---|---|---|---|---|
| `fork_stl` | ForkMilimeter.stl (mm) | fork | 9.48 g | -24.3, 0.0 mm | yes (0.000 mm) | yes | tines 7 / palm 1 / neck -15 / handle | 0.01, 17 parts |
| `fork2_stl` | Fork.stl (m) | fork | 13.41 g | -21.3, 0.0 mm | yes (0.000 mm) | no (flat, symmetric) | tines 7 / palm 1 / neck -15 / handle | 0.01, 7 parts |
| `knife_stl` | Knife.stl (m) | knife | 16.21 g | -8.1, -1.4 mm | **no** (7.9 mm) | yes (mirror images) | blade 1 / bolster -15 / handle | 0.01, 2 parts |
| `spoon_stl` | Spoon.stl (m) | spoon | 21.66 g | -2.9, 0.0 mm | yes (0.123 mm) | no (flat, symmetric) | bowl 1 / neck -15 / handle | 0.01, 4 parts |

* **The registry** holds, per object: the STL and its units (and why), the scene, the mesh folder, material and
  density (steel 7850 kg/m³; no part of any mesh suggests another material: each is one solid of uniform thickness),
  the mass and centre of mass computed from the mesh, the CoACD threshold, the mirror symmetry, whether "face up /
  down" means anything, the parts along the long axis and the stable poses. Order = the object id of the models.
* **fork_stl moved into the registry first.** `collect_dataset.py` (new `use_object(name, scene)`) and
  `train_centre.py` read its name, mesh folder and part boundaries from it. Check before and after
  (`results_multi/registry_check/`): 40 regenerated dataset_mini presses and the final model's 20 stored search
  episodes are identical, with the same digests.
* **The new files are in metres.** Their longest extents are 0.120 / 0.144 / 0.129 file units, and only metres give a
  cutlery length (120-250 mm). They are scaled to mm copies (`results_multi/objects/source_mm/`) and imported with
  `stl_to_mjcf.py --units mm`, like fork_stl (the whole pipeline assumes mesh files in mm).
* **All three new objects are flat 2 mm plates with the same handle**, which is also fork_stl's handle (width
  4.08 - 0.050 x mm from -84 to 0 mm). Fork.stl has fork_stl's exact top-view outline (4 tines from x = 7 mm), but it
  is flat, while fork_stl is a bent 1.5 mm sheet 5 mm tall. The spoon's "bowl" is a flat oval with no hollow; the
  knife's blade widens to one side only.
  * So one press on the handle cannot tell Fork / Knife / Spoon apart, and the centre of mass seen from the handle
    depends on the object (x = -21.3 / -8.1 / -2.9 mm): this is what the joint search has to resolve.
* **CoACD threshold, by a rule** (`results_multi/objects/import_objects.py`; the rule was revised twice before the
  choice, each time logged with its reason):
  * thin features kept: forks have 4 tines and all three slots open along the whole tines (narrowing <= 0.1 mm); the
    parts stay within 0.15 mm of the knife's cutting edge and the spoon's rim, 0.25 mm at concave corners, 0.2 mm in 3-D,
    top-view IoU >= 0.99;
  * and contact physics: a static press on every site at 0.5 / 2 / 5 / 10 N within max(5 %, 0.03 N).
  * The rule's second version had chosen 0.0025 for the fork (32 parts); its handle, cut differently from the knife's
    and spoon's, pressed 17 % over the command. A handle cut differently on one object would be a simulation
    artifact the network could recognise the object by. With 0.01 the handle is one identical convex part in all three
    (x -84..-0.2 mm, 36 vertices, 1008 mm³).
* **verify_stl.py** (`results_multi/objects/verify_<name>/`, galleries looked at): rest, overlay (3-D <= 0.18 mm),
  footprint IoU 0.999 / 0.999 / 0.9998 (100 % of contact points on the visual footprint), the fork's slot probes
  (CoACD within 0.004-0.009 mm of exact first contact; the plain convex hull would be 0.14-0.32 mm off), SDF
  comparison. The checks that fail were looked at one by one and none is a defect of the welded objects the pipeline
  uses: a 2.4 µm jitter limit (force 4.1 % <= 5 %, identical on all three), float32 round-off (0.13 µm), the known
  free-object jitter (`verify_stl.KNOWN_FREE_ISSUE`), a tilted pad touching the table beside a 2 mm thin handle, and a
  1e-6 N summation-order difference of the primitive spoon in the bigger scene. All three objects are kept.
* **Stable poses** (`dataset_multi/poses/<name>/`, fork_stl's rules: 1 mm release, 20 tilted releases, 500 drops,
  natural = all pass and >= 1 % of drops): Fork 2 natural (flat) / 4 propped; Spoon 4 natural (flat x 2, standing on
  its edge x 2: 1.8 / 2.0 % of drops) / 1 propped; Knife 4 natural (flat x 2, on its straight back 1.6 %, on its cutting
  edge 2.0 %). Propped poses are not sampled.
* **Scene:** `gelsight_multi.xml` includes the unchanged `gelsight_moving.xml` and the three objects. fork_stl's presses
  are identical in both scenes (15 / 15 seeds).

### 2. Data (`collect_multi.py`, `dataset_multi/`)

* dataset_mini's settings (Mini sensor, welded object, natural poses, 70 / 20 / 10 % on the object / table / hovering,
  0.2-10 N, tilt <= 3 deg); **per object 30k train / 2k val / 3k test** (140,000 presses; 23 min on 24 CPU workers).
* **A shard keeps only what the network uses:** X (the reconstructed depth at the model's 72 x 93 px, float16, the same
  transform as dataset_mini's cache: equal on 200 presses), for train presses the 5 structured-noise variants the
  training draws from, all labels plus object id / part / face, the contact rows, the seeds and per shard the reference
  image. The RGB image, the full-resolution depth maps and the markers are rebuilt from the seed (`collect_multi.py
  regen`: bit-identical on the same EGL device).
* **Disk: 545 MB per 10k train presses (with the bank), 110 MB per 10k val / test presses**; `dataset_multi/` 6.3 GB,
  plus the training cache (1.9 GB) and bank (8.0 GB).
* **Label check** (`results_multi/data_check/label_gallery_<name>.png`): the object outline drawn from the labels lies on
  the depth signal for all four objects. Presses in the middle of the flat bowl give a weak, almost uniform depth
  image (photometric reconstruction sees slopes, not height); a press flat on the table can be completely dark.

### 3. Models (`centre_multi.py`)

* **A "multi-all"**: the four objects (120k train presses). **B "multi-holdout"**: fork_stl, Knife, Spoon only (90k);
  Fork.stl is never seen in training.
* **The final fork model's recipe**: the same backbone, training (120 epochs, AdamW, OneCycle, bf16), inputs (X + the
  bank), robust augmentation and test border. Mirror augmentation only for presses of mirror-symmetric objects.
* **Heads**: contact (hover / table / object); identity (softmax over the model's objects); part (one block of logits
  per object, the loss uses the true object's block); face (only where defined); the centre-of-mass heatmap (150 x 150
  cells of 2 mm, as before).
* **One heatmap decoder shared by all objects, conditioned on the object hypothesis**: it is fed the 512-d code plus a
  learned embedding of the object. Training uses the true object; inference runs the decoder once per hypothesis, so
  the search gets P(centre | press, object = h) for every h. A single unconditioned heatmap could not give a belief per
  hypothesis: on the shared handle it would mix the four objects' centres. The identity-weighted mixture of the
  hypotheses' heatmaps is the "generic" heatmap.

### 4. Search with an unknown object (`search_multi.py`)

* The v2 loop of the fork search (level sensor after the first press, backtracking after an empty press), with a
  **joint belief over identity and centre**: log P(h, c) = log w_h + the sum of each contact press's heatmap for
  hypothesis h (1 % uniform mixing, as before); w_h multiplies the identity head's outputs (1 % mixing). An empty press
  takes the whole footprint out under every hypothesis. P(h) sums P(h, c) over c, so it also counts how well the presses
  agree on a centre under h.
* **Next press**: the largest expected information gain about (h, c), the sum of the centre part (the fork search's
  model on the marginal centre belief) and an identity part (under hypothesis h the identifying head lies d_h from the
  centre along the axis; a press that would land on it reveals h). The sum is an approximation of the joint gain.
* **Stop**: the most likely object has >= 95 % and its own 90 % region is below 16 mm², or 10 presses.
* **Oracle**: the same loop with only the true object's hypothesis (the identity given). **Generic** (model B on the
  unseen fork): one pseudo-hypothesis, the identity-weighted mixture.
* Starts: 300 per object, on-object test presses with contact (all parts), from `dataset_multi/<object>/test-0000.h5`.

### 5. The full pose (pose addendum): conventions, heads, joint belief, CAD check

The goal became the full pose of the object on the table: centre of mass (x, y), yaw, stable-pose id, and which object.
The centre of mass stays the reference point (it is also where a robot should grasp).

**Geometry and printing** (`results_multi/objects/geometry_check.py`, `printability.py`):
* The trained geometry is the printed geometry: the visual meshes have exactly the original STL triangles (matched one by
  one, no flipped faces); the largest vertex deviation is 2.9e-6 mm (float32 rounding of the mm copy), MuJoCo's compiled
  mesh is within 5.9e-6 mm. No repair, decimation or smoothing was needed (all four are watertight, 152-986 triangles).
  `objects.json` holds the SHA-256 of every original STL and of the mm copies.
* **Fork.stl, Knife.stl and Spoon.stl are in metres**: a slicer that assumes mm prints them 1000 times too small. The
  trained size is the file x 1000 (120 / 144 / 129 mm); the mm copies in `results_multi/objects/source_mm/` can be
  printed as they are.
* Printability (geometry not modified): thinnest features ForkMilimeter.stl 0.91 mm (the sheet on the tines, x 15-33 mm),
  Fork.stl 1.02 mm (tine ends), Knife.stl 1.24 mm (rounded tip), Spoon.stl 2.0 mm. Nothing under 0.8 mm.

**Conventions** (`objects.json` "conventions"): object frame at the centre of mass, +x along the long axis towards the
head; yaw = the angle of +x about the vertical (table frame; the network predicts it in the sensor frame); stable-pose ids
= the object's own. Indistinguishable pose pairs: Fork.stl 0 / 1 and Spoon.stl 0 / 1 (flat, symmetric about the long axis
and the mid-plane: face up = face down), Spoon.stl 2 / 3 (either edge). The label check
(`results_multi/data_check/pose_label_check.py`): the CAD outline placed only from the labelled pose lies on the depth for
30 presses per object; the yaw label agrees with the image's axis to 0.9 deg (1.9 deg after augmentation; 45.6 deg if
the mirror were left out of the label transform).

**Heads** (`centre_multi.py`): yaw = 72 bins of 5 deg in the sensor frame (cross-entropy; the label is transformed by the
augmentation's rotation / mirror like the image); stable pose = one block of logits per object over its natural poses.
The first model A (no pose heads) was stopped at epoch 10 and kept in `results_multi/superseded/`.

**Joint belief on the GPU** (`pose_search.py`): (object, stable pose) x 72 yaw bins x a 201 x 201 mm centre grid (1 mm)
in the episode's sensor frame: 140 MB float32 for model A's 12 (object, pose) pairs, plus 42 MB of touch masks; measured
peak 0.50 GB per search process (three run next to a training). The yaw grid is the head's own (5 deg). The heads are
combined as a factorised likelihood - **this assumes they are independent given the pose**, which they are not exactly
(they share the image). A press also tests the object's real footprint at each pose against the pad (contact needs a
touch; an empty press rules out every pose that would have been touched). Information gain is over the whole pose: 2,048
states sampled from the belief predict what the pad would see at each candidate press (nothing, or which region of which
object; the shared handle of Fork / Knife / Spoon is one region class), and the gain is the entropy of those outcomes.
Stop: object >= 95 %, centre 90 % region < 16 mm², yaw 90 % interval within +-3 deg, stable pose >= 95 % (an
indistinguishable pair counts as one); or 10 presses.

**CAD check and refinement** (C4, no training): the top 64 cells after each contact press are rendered from the CAD
(expected indentation from the top-surface height at that pose) and compared with the reconstructed depth (masked mean
squared error, summed over the presses). At the end the best pose is refined continuously (x, y, yaw).
* A first temperature (tau = 0.033, calibrated on the CAD error alone) was far too sharp: after one press on the knife's
  blade the belief went to the mirrored stable pose with P = 1.0 and the search stopped wrong. Calibrated GIVEN the network
  (`pose_calib_given_net.py`) the CAD still adds sub-bin information (log-probability of the true pose among 25 nearby
  candidates -2.13 with the network alone, -1.23 with the CAD at tau = 0.027), but this does not see the search's selection
  bias (the best of 64 cells from a large grid can beat the truth by chance).
* By a rule written before the results, the check's temperature was chosen on 160 validation searches
  (`choose_tau.py`): **tau_search = 1.0** (94.4 % success at ADD < 5 mm, 0.6 % confidently wrong stops; tau 0.1: 95.0 % /
  3.1 %).
* The final refinement uses tau = 0.027 with the network belief as prior: a decision taken after seeing the validation runs
  (stated as such in the run log): the refinement is a local comparison of nearby poses given the network, which is what
  that calibration measured. The refinement on the CAD match alone is kept per episode for comparison.
