### Real-sensor tooling (prepared, not run on hardware)

Three scripts for a GelSight Mini on a robot. They are written against the Mini's standard video
interface: a UVC camera read with OpenCV (V4L2 on Linux).

> **None of this has touched a real sensor or robot.**
> * The analysis parts of `calibrate_real.py` and `measure_noise.py` were run on simulated
>   sessions with known ground truth (`--simulate`).
> * The capture code (`record_real.py`) and the robot interfaces have not been run at all.

| script | does | tested |
|---|---|---|
| `record_real.py` | finds the camera whose V4L2 name contains "GelSight Mini" (or `--device N`); grabs MJPG frames (requests 3280x2464, uses what it gets); crops 1/7 of each side and resizes to 320x240 like gsrobotics' `resize_crop_mini`; saves reference frames (mean of `--ref-avg`), press frames, and per press the robot pose (`--robot manual`, `file:PATH` JSON written by your robot program, or `ur:HOST` via ur_rtde); `--noise-frames N --interval s` records no-contact frames over time for `measure_noise.py` | **no** (no hardware) |
| `calibrate_real.py` | mm/px from a recorded session; result in `<session>/scale_calibration.json`, plus the resampling factor the networks need (they assume 0.1 mm/px). Two methods: `gauge` (two balls a known distance apart, median of >= 5 presses, presses with unequal contact areas rejected) or `displacement` (one ball, >= 6 known robot offsets, affine fit, which also reports the anisotropy). Centroids come from the RGB colour change, so no photometric calibration is needed first | analysis only, on simulated sessions: gauge, true scale 1.02, error -0.04 %; displacement, true scale 0.97 (robot noise 0.02 mm), error +0.19 %, anisotropy 1.0005 |
| `measure_noise.py` | from no-contact frames over time, in the augmentation's units: camera noise (grey levels); the same noise after the photometric pipeline (mm per input pixel, vs the augmentation's 0-0.05 mm); illumination drift as the max of a quadratic gain fit per channel (vs 0.01-0.08), plus its rate per minute; static texture (upper bound only); marker residue (lower bound only) | analysis only, on simulated frames with injected camera noise 1.5 grey, drift ramp to 4 %, texture 2 grey: measured 1.52 grey, drift 0.036 (p95 0.035), texture "<= 2.0" (it reads 1.6 with no texture at all, so it is only an upper bound) |

**Using the scripts on the real sensor:**
1. Record a gauge session and run `calibrate_real.py` -> mm/px.
2. Calibrate the photometric model with a ball at that scale
   (`fork_detector.PhotometricModel.calibrate_ball`). The default `results_moving` model was
   calibrated on the simulator.
3. Record `--noise-frames` and run `measure_noise.py` with that model.
4. Replace the augmentation ranges in `centre_v2.py` (`random_variant`, `augment_v2`) with the
   measured values.

A useful by-product of the self-test: the simulator's own camera noise (1.5 grey levels) already
gives about 0.03 mm of per-pixel depth noise at the network input. The augmentation's 0-0.05 mm
range is the same order.

**What cannot be measured without contact:** how much the gel texture shifts against the
reference under a press, and the real marker residue (markers do not move without contact).
Both need pressed frames of a known object.

**Device assumptions to check first:**
* the camera name;
* the stream resolution and whether MJPG is offered;
* that the 1/7 crop removes this unit's black border;
* the image handedness vs the simulator (image column = sensor +x; `--flip-x` / `--flip-y`);
* whether auto exposure can be locked (`--lock-exposure`).

A drifting exposure is exactly the illumination drift that breaks the old contact head.
