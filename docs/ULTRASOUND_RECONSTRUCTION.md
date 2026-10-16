# Reconstruction from tracked ultrasound

The current live preset uses ReMIND-041 (2.257 mL, about 19 × 18 × 18 mm), selected
from actual DICOM SEG voxel volumes and checked against the public clinical
workbook. Its reported diagnosis is astrocytoma and 95.6% of sampled tumour voxels
overlap the cerebrum in physical coordinates. It is not a scaled version of
ReMIND-001. The source collection is
[TCIA ReMIND](https://www.cancerimagingarchive.net/collection/remind/); selection
results and the public clinical workbook are cached under `data/remind_case_search`.

Brain, tumour and cavity acoustic support now use the same smoothed MRI 0.5
isosurfaces as the display geometry. Scatterers are sampled in that continuous
anatomy, rather than coarse occupied FEM cubes, and embedded in FEM tetrahedra.
Detailed display skins now use tetrahedral material coordinates too. Samples
outside the coarse exterior extrapolate the nearby FEM field; this is a display
and acoustic-domain approximation, not a finer mechanical solve. In ReMIND-041,
the old coarse tumour contour differed from its detailed MRI contour by about
1.51 mm in the resting central slice. The new material field agrees with MRI
surface vertices to less than 6.2e-7 in isovalue (dimensionless); this is a boundary
consistency check, not a claim of ultrasound localization accuracy.

With fine MRI sampling, approximately 8.8% of the current scatterers extrapolate
outside the coarse exterior. This is recorded in the episode reference. It
preserves the visible/acoustic anatomy but does not improve the resolution of
the mechanical collision mesh.

The live image uses 256 × 256 samples, 40 mm lateral coverage and depths 2–80 mm.
The 80 mm depth contains the full smaller tumour. Calibration travels with every
image and controls reconstruction, contour scoring and browser aspect ratio.
The actual probe footprint determines clearance on each anatomy; fixed scan
limits are clipped to supported tissue. The physics ground-truth panel continues
to render conventional triangle meshes, without Gaussian splats.

The implementation has two distinct inverse outputs: a 3D echo-intensity volume
from tracked B-mode images, and incremental deformation of an initial MRI mesh
from image motion. The coloured brain, tumour outline and blue ventricles are
initial MRI priors, not ultrasound segmentations. The **Echo volume · ultrasound
only** layer displays the image-derived volume without those anatomical surfaces.

The live `/comparison.html` view places an ultrasound-updated tumour on a fixed
brain beside a conventional shaded triangle mesh of actual Isaac GPU FEM
deformation. The ground-truth panel has no Gaussian representation. Detailed MRI
skins are reduced by vertex clustering for display (brain 1 mm, tumour 0.6 mm,
ventricles 1.2 mm); averaged sparse skin bindings transport those triangle vertices
with actual simulator nodal displacement. The native Isaac application runs
headless. The mesh panel renders the simulator's measured state in WebGL.

The evaluation worker separately consumes versioned simulator snapshots and the
already-produced inferred nodes. It checks epoch, frame and timestamp before
publishing a complete image/prediction/truth set. The reconstruction worker never
reads those truth snapshots. Initial, inferred and actual full-resolution tumour
skins are intersected with the calibrated ultrasound plane. Their outlines are
overlaid on the observed image, on both 3D views and in a spatially zoomed plot.
No motion magnification is applied. Slice Dice uses the acquired 192×192 image
grid; symmetric contour distances use boundary samples at approximately 0.1 mm
spacing within the image field; whole-tumour RMSE uses corresponding surface
vertices, not nearest-surface distances. Absent truth contour coverage is reported
as unavailable rather than a perfect score. All comparison metrics are synthetic
evaluation diagnostics, not clinical accuracy claims.

The earlier ReMIND-001 paired-view integration check recorded 24 changing comparison frames with
matching image/prediction/truth IDs and no browser or WebGL errors. The actual
tumour surface moved approximately 1.73 mm RMS from its initial shape at peak;
the final sample measured about 1.0 inverse update/s and 28 viewer frames/s.
The normal ground-truth display contains 223,478 indexed mesh vertices. The
recorded browser video and checks are under `outputs/tumour_comparison`.
The current smaller-case measurements are given below.

## Smaller-case replay and independent validation

The new pipeline was developed on 135 observations from two actual Isaac press
cycles with scatterer seed 42. Sparse LK tracking plus a stronger elastic prior
(data weight 10 rather than 5000) and rejection rollback reduced mean contour
distance from 0.510 mm to 0.311 mm. Those observations were used to select the
regularization weight, so they are development results.

A separate 128-frame scan used seed 99, continuous MRI brain/cavity scattering,
and a different lateral/depth trajectory. Parameters were frozen before that
scan. Both pipelines replayed exactly the same images and calibrated poses;
ground-truth nodal files were opened only in the subsequent scoring function.

| Method | Mean contour distance | Mean 3D surface vertex RMSE | Mean slice Dice | Accepted updates |
| --- | ---: | ---: | ---: | ---: |
| Static initial MRI tumour | 0.970 mm | 1.377 mm | 0.872 | baseline |
| Previous live pipeline: TV-L1, weight 5000 | 0.447 mm | 1.172 mm | 0.939 | 28.1% |
| Updated pipeline: sparse LK, weight 10, rollback | 0.330 mm | 0.710 mm | 0.956 | 89.8% |

This is about 26% lower contour error and 39% lower surface RMSE than the previous
pipeline on that scan. It is a limited synthetic test on the same patient prior,
not a real-ultrasound or cross-patient accuracy claim. The baseline replay
processes every frame offline, despite being too slow for the live acquisition
rate. Mean flow/inverse time was 1372 ms for the baseline and 56 ms for the
updated pipeline, excluding one-time elastic precomputation and live volume,
transport and browser work. The final paired browser check measured about
3–4 inverse updates/s and 32 display frames/s, with no browser/WebGL errors,
24 changing matched frames and a normal triangle-mesh ground-truth renderer.
Rejected updates still retain the previous estimate; out-of-plane motion remains
prior-driven.

```bash
# While a fresh live run is active, optionally using --seed 99:
.venv-ultrasound/bin/python scripts/record_tracking_benchmark.py --profile validation --seconds 32
# Substitute the directory printed by the live launcher:
OPENBLAS_NUM_THREADS=1 .venv-ultrasound/bin/python scripts/benchmark_ultrasound_tracking.py \
  outputs/live_<timestamp> --output outputs/small_tumour_validation
DISPLAY=:1 .venv-ultrasound/bin/python scripts/check_tumour_comparison.py \
  --output outputs/small_tumour_comparison_final
```

The evaluation JSON is under `outputs/small_tumour_validation`; the current
video is `outputs/small_tumour_comparison_final/small_tumour_comparison.mp4`.
Older ReMIND-001 validation and RF recordings below are retained as historical
baselines and are not the current live default.

## Observation path and input separation

`scripts/reconstruct_ultrasound.py` reads `iq`, `probe_position`,
`probe_rotation` and `time` from an acoustic directory, plus its acquisition
calibration and fixed display gain. It does not read the episode's `state/`,
loading command, deformed surface or reference tumour mask. The initial anatomy
and material Gaussian files are explicit prior inputs. Source hashes and input
roles are saved in `manifest.json`.

The live forward worker reads the actual simulator scatterers to **generate**
an observation. A separate inverse worker reads only the resulting image/pose
NPZ and initial mesh prior. This separates privileged forward-simulation state
from reconstruction measurements. Epoch and frame identifiers synchronize reset
and image timestamps; they do not encode tissue deformation. Versioned image and
estimate files prevent a newer estimate from being paired with an older image.

The live ultrasound source is an approximate coherent beamformed-IQ PSF model:
signed scatterer coefficients, propagation phase, elevation sensitivity, axial
and lateral Gaussian resolution, envelope detection and a fixed 60 dB display
range. This is not channel RF or full wave propagation. The recorded page uses
the existing SIMUS RF-derived images instead. A realistic-looking preview alone
does not validate either model against a scanner.

## Tracked 3D echo reconstruction

Each volume voxel is transformed into each calibrated probe plane. Bilinear
in-plane image sampling and a Gaussian elevation kernel accumulate weighted
intensity and coverage. The recorded preset uses a 1 mm grid, 1.5 mm elevation
sigma and truncation at three sigmas. Unobserved voxels retain zero coverage;
they are not treated as measured black tissue or filled with MRI labels.
The renderer represents observed voxels as isotropic Gaussian footprints.
It does not optimise an ultrasound Gaussian neural scene or discover semantics.
The volume compounds different acquisition times while tissue is deforming,
so its texture can blur and its geometry is not a motion-corrected 4D volume.

Every fifth frame (indices 2, 7, 12, 17, 22, 27 in this scan) is excluded from a
separate validation volume. Reconstructed held-out slices are scored only where
there is support from other images. The current sparse sweep gives zero support
for held-out frames 7 and 22. Covered foreground in frames 12, 17 and 27 yields
approximately 19.5–19.9 dB PSNR. Frame 2 has no foreground. Those numbers do not
represent successful recovery of all missing slices or all anatomy.

## Image-driven elastic inverse

Only nearly coplanar observations are accepted for motion tracking. Calibrated
probe motion is compensated before image flow is measured. Existing recorded
RF reconstruction uses TV-L1. The live default uses OpenCV pyramidal Lucas–Kanade
at the material points, with forward/backward consistency and normalized patch
correlation. High-confidence matches to the reference image correct accumulated
motion. A rejected elastic proposal does not become the next cumulative target.
Adjacent-pair
tracking samples flow at prior-predicted material locations; forward/backward
consistency determines measurement reliability. Unreliable observations lose
weight rather than becoming artificial zero-displacement constraints.

The initial conforming tetrahedral mesh supplies a small-strain isotropic
elastic stiffness matrix, including its heterogeneous assumed regional moduli
and a fixed base. Shared material-interface nodes are unified. The inverse solves

`min_u 0.5 u^T K u + 0.5 alpha ||W^(1/2) (C u - d_US)||^2`.

`C` maps nodal displacement to measured lateral/depth motion at image material
points. Normal displacement is unmeasured and follows the elastic prior. The
data weight is an assumed regularization choice, not a calibrated force or
uncertainty model. The live data weight is 10, selected on the development replay
to reduce overfitting of decorrelated speckle; the legacy baseline uses 5000.
Sparse factorization and precomputed elastic compliance
reduce each subsequent update to a small confidence-weighted Schur solve.
The resulting nodes move the prior Gaussians, including `F Sigma_0 F^T`
covariance transport. No deformation state or prescribed probe load is fitted.

Strong return to the reference image appearance can close a tracking loop.
Updates exceeding 4 mm incremental nodal motion or a volume ratio below 0.3 are
rejected, retaining the last valid estimate and reporting the condition. These
are prototype admissibility bounds, not evidence that the estimate is correct.
Reset establishes a new image reference. Changes to the scan plane invalidate
the local motion observation; the system does not fabricate dense 3D flow for
arbitrary freehand motion.

## Evaluation and practical limits

`scripts/evaluate_ultrasound_reconstruction.py` is a separate **evaluation-only**
path that can read synthetic ground-truth states after reconstruction. It scores
incremental in-plane displacement in the imaged slab against a zero-increment
baseline. The current example improves compression-phase error but can be worse
than the zero-motion baseline during release: speckle decorrelation, accumulated
tracking error and incomplete observations remain important limitations.
For example, one compression frame has about 0.19 mm inverse RMSE versus
0.35 mm for zero increment, whereas a release frame has about 0.38 mm inverse
RMSE versus 0.13 mm for zero increment. These are limited synthetic diagnostics,
not dense whole-brain or clinical validation.

The recorded build takes about 33 seconds here, including initial sparse
factorization/compliance computation. Subsequent image/inverse/transport updates
are approximately 0.7 seconds per frame. Playback and browser rendering can run
faster; they are not the inference rate. Live image age and inverse rate remain
visible. The desktop launcher disables GPU presentation throttling observed on
this display, while the renderer caps drawing at 60 frames/s.

The recorded/live browser integration check passed with no WebGL or page
errors. It measured approximately 41 frames/s in the recorded reconstruction
view and 45–48 frames/s in the live paired view. The live inverse published at
about 1.1 Hz during pressing, with a matched image/estimate age around 1.8 s;
its frame IDs were checked for equality. These rates are different quantities.
Evidence and video are under `outputs/ultrasound_reconstruction/final_video`
and `outputs/ultrasound_reconstruction/ultrasound_reconstruction.mp4`.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-ultrasound/bin/python \
  -m unittest discover -s tests -v
.venv-ultrasound/bin/python scripts/evaluate_ultrasound_reconstruction.py \
  outputs/ultrasound_reconstruction --truth-episode outputs/tumor_scan_contrast
DISPLAY=:1 .venv-ultrasound/bin/python scripts/check_ultrasound_viewer.py
```

Tests include physical calibration, unknown coverage, probe-motion compensation,
known image translation, elasticity rigid modes, image-constraint fitting,
confidence weighting, coherent signal cancellation and point-target localization.
The browser check records the actual image/geometry flows and verifies that an
estimated mesh is paired with the frame used by its inverse.

## Research relationship

Tracked slice compounding is a standard freehand ultrasound reconstruction
baseline; see [Huang et al., kernel-regression reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC4165991/).
[UltraGauss](https://www.robots.ox.ac.uk/~vgg/research/UltraGauss/) motivates
ultrasound-specific plane sampling rather than treating B-mode as an optical
camera image. Its official page currently marks code as coming soon. This
implementation is independent; it does not reproduce UltraGauss or the learned
time-evolving stress/velocity field of [PIDG](https://arxiv.org/abs/2511.06299).
Learning a 4D acoustic/material field, calibrated uncertainty and robust recovery
from real tracked ultrasound remain research work.
