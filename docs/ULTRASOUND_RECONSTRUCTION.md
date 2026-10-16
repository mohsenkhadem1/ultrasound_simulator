# Reconstruction from tracked ultrasound

The implementation has two distinct inverse outputs: a 3D echo-intensity volume
from tracked B-mode images, and incremental deformation of an initial MRI mesh
from image motion. The coloured brain, tumour outline and blue ventricles are
initial MRI priors, not ultrasound segmentations. The **Echo volume · ultrasound
only** layer displays the image-derived volume without those anatomical surfaces.

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
probe motion is compensated before TV-L1 image flow is measured. Adjacent-pair
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
uncertainty model. Sparse factorization and precomputed elastic compliance
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
