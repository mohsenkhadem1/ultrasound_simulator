# Isaac Sim deformable brain and tissue with acoustic ultrasound

The live default is now **ReMIND-041**, a measured 2.257 mL tumour with dimensions
approximately 19 × 18 × 18 mm. The earlier ReMIND-001 lesion was about 27 mL.
This is a different patient segmentation, preserved at its physical scale.
The selection checks source voxel volume, clinical whole-tumour volume,
histopathology and physical tumour/brain overlap; it excludes inflammatory
non-tumour cases and ambiguous small masks that disagree with whole-tumour volume.
Source: [TCIA ReMIND](https://www.cancerimagingarchive.net/collection/remind/),
CC BY 4.0. Source series and hashes are saved in each case's `provenance.json`.

Live ultrasound now samples the detailed MRI brain, tumour and ventricular
boundaries, with persistent material scatterers transported by FEM tetrahedral
coordinates. This removes the previous acoustic 5 mm block boundaries while
keeping the physics mesh coarse. Fine exterior samples can extrapolate outside
that coarse mesh, just as the MRI display skins do. The live field extends to
80 mm depth on a 256 × 256 grid, and all slice overlays use its metric calibration.
The probe clearance uses its actual head footprint instead of a broad scan envelope.

The live estimator uses pose-compensated pyramidal Lucas–Kanade tracking,
forward/backward and patch-correlation checks, strong reference-image matches,
and an elastic inverse with data weight 10. Rejected targets do not accumulate.
This replaces the slower live TV-L1 tracker; TV-L1 remains available as a replay
baseline and for existing recorded RF reconstruction. Synthetic validation and
the remaining limitations are recorded in
[the reconstruction notes](docs/ULTRASOUND_RECONSTRUCTION.md).

On a separate 128-frame synthetic scan with a different speckle seed and probe
trajectory, mean tumour contour error fell from **0.447 to 0.330 mm**, and mean
3D tumour surface RMSE from **1.172 to 0.710 mm**, compared with the previous
pipeline on identical observations. These are same-patient synthetic diagnostics;
out-of-plane motion and some rejected updates remain limitations. The current
paired view measured about **3–4 inverse updates/s and 32 display frames/s**.

## Side-by-side tumour prediction and Isaac ground truth

Run `bash scripts/run_live_brain.sh`, then open
`http://127.0.0.1:8765/comparison.html` or run
`.venv-ultrasound/bin/python scripts/open_live_viewer.py --comparison`.
The left panel keeps the brain fixed and moves the tumour using the ultrasound
image-driven elastic inverse. The right panel is a **conventional shaded triangle
mesh**, transported by actual Isaac GPU FEM nodes at the same ultrasound
timestamp. Isaac runs headless; the right panel renders its measured state rather
than embedding the native Isaac viewport. It uses no Gaussian splats.

**Start moving probe + press** translates the probe within the imaging plane and
presses/releases through 0–4 mm commanded travel over a 16-second cycle. Both
cameras orbit and zoom together. The observed ultrasound and a zoomed contour
plot overlay the initial MRI tumour (dashed amber), ultrasound-updated MRI tumour
(cyan) and Isaac ground truth (pink). Slice Dice, mean contour distance and
whole-tumour corresponding-vertex RMSE compare both predictions with truth.
All geometry is shown at actual scale; the close-up uses spatial zoom.

A separate evaluation worker reads actual Isaac snapshots **after prediction**.
It requires matching epoch, frame and image timestamp. Truth never enters the
inverse observations. Full-resolution triangle/plane intersections supply the
contours; display mesh decimation does not affect the contour scores. Predictions
still begin with the MRI tumour prior and can drift or be worse than the initial
contour. This is deformation tracking, not ultrasound tumour segmentation.
Live images use the coherent-IQ PSF preview described below.

```bash
DISPLAY=:1 .venv-ultrasound/bin/python scripts/check_tumour_comparison.py
```

This check records the actual browser flow and verifies synchronized frames,
changing predicted/actual tumour geometry, conventional ground-truth mesh
rendering and WebGL errors. Evidence is saved under `outputs/tumour_comparison`.

## Ultrasound observations and 3D reconstruction

The browser now displays the corresponding ultrasound image beside the 3D
model. **Geometry source** switches between an ultrasound-driven estimate and
the simulator reference. **Start fixed-plane press** keeps the plane observable
while the probe moves within the plane and presses/releases through a 4 mm commanded stroke.
Moving to a different plane resets local
motion observability. Image timestamps, image age and inverse update rate are
displayed separately from the physics and viewer rates.

There are two acoustic sources:

- The live preview uses a fast coherent-IQ point-spread model of the actual
  advected scatterers, including coherent speckle and finite resolution. It
  approximates beamformed IQ; it does not calculate channel RF.
- The recorded reconstruction uses the existing **SIMUS RF → IQ → DAS B-mode**
  scan. The reconstruction page shows the observed image, reconstructed reslice,
  image-motion arrows, an image-only echo volume, and an ultrasound-updated MRI
  mesh prior. Its playback is explicitly a recorded synthetic acquisition.

```bash
# Build from the existing tracked acoustic scan (no deformed states read):
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-ultrasound/bin/python \
  scripts/reconstruct_ultrasound.py

# Run live physics, ultrasound preview and the observation-only inverse worker:
bash scripts/run_live_brain.sh
# In another terminal, open the live image + estimate:
.venv-ultrasound/bin/python scripts/open_live_viewer.py
# Or the RF-derived scan reconstruction:
.venv-ultrasound/bin/python scripts/open_live_viewer.py --reconstruction
```

To explicitly use the earlier patient, set `DEFORM_ANATOMY_DIR=data/remind-001`
and `DEFORM_PATIENT=ReMIND-001`. Legacy recorded RF reconstruction is served only
when `DEFORM_RECONSTRUCTION_DIR=outputs/ultrasound_reconstruction` is explicitly
set, so it is not silently mixed with the smaller live case.

The reconstruction uses image intensities, calibrated pixel coordinates,
tracked probe poses and an **initial** elastic mesh. It does not ingest the
simulator's deformed tissue state or loading command. The MRI tumour outline
remains an input prior: this does not segment a new tumour from ultrasound.
Only in-plane image motion is measured; the elastic prior determines the 3D
extension. Tracking drift and release-phase error remain substantial. See
[method, timing and independent evaluation](docs/ULTRASOUND_RECONSTRUCTION.md).
The first elastic precomputation takes tens of seconds; subsequent inverse
updates are slower than the physics loop. Reconstruction exports include a
voxel intensity/coverage NPZ, inferred states, source hashes and held-out metrics.

## Live 3D Gaussian brain and tumour

The live viewer uses the patient mesh prior to initialise 45,192 anisotropic
Gaussians. Actual Isaac GPU contact physics moves their material coordinates;
each soft Gaussian's covariance follows `F Sigma_0 F^T`, including compression
and shear. The browser projects full covariances and depth-sorts alpha-composited
splats using WebGL2. No prerecorded motion is used.

```bash
# After installing and preparing the ReMIND anatomy as described below:
bash scripts/run_live_brain.sh
# Open http://127.0.0.1:8765 in a WebGL2-capable browser.
# Optional automatic probe motion:
bash scripts/run_live_brain.sh --live-autoplay
```

Move and turn the probe with the sliders, push through 0–5 mm of commanded
travel, or start the automatic scan. Pause and reset operate on the simulation.
Automatic scanning uses a gentler maximum 3 mm travel, follows terrain at the
probe's actual position, and takes 12 seconds per pass.
Orbit with the mouse and zoom with the scroll wheel. The view includes a brain
cutaway, isolated tumour, displacement and stiffness colours, and an analytic
stress diagnostic. Motion magnification is explicitly a display option; the
default is actual scale. Blue structures are ventricles; the cyan rectangle
marks the nominal acoustic plane, not a reconstructed image.

The live default uses 90 Hz contact substeps and targets 30 Gaussian updates/s.
The viewer displays measured stream and render rates separately. Use
`--live-physics-hz 240` for the offline episode's smaller timestep at the cost of
speed; the value must be a multiple of `--live-fps`. Rates depend on hardware and
load. A local HTTP/WebSocket server binds only to `127.0.0.1`. Press Ctrl+C in
the launching terminal to stop. Live outputs include metrics, metadata and a
final nodal state. The current-image workers write bounded live image/estimate
buffers rather than a complete RF acoustic episode.
Every update checks volume ratios across the full FEM mesh and pauses physics
if it detects an inverted element; the viewer displays that condition and offers reset.

This is an adaptation inspired by
[Physics-Informed Deformable Gaussian Splatting](https://arxiv.org/abs/2511.06299),
using an explicit FEM/material prior. **It is not a reproduction of the paper's
trained velocity/stress field or optical reconstruction system**. The new image
motion / elastic inverse is described above; see also
[the PIDG method mapping](docs/PIDG_ADAPTATION.md). Full acoustic RF simulation
remains the separate offline process below.

To regenerate the prior, run `.venv-ultrasound/bin/python scripts/prepare_gaussians.py`.
The Gaussian server uses Isaac's existing `aiohttp` and Warp packages. The
optional browser integration check requires Playwright 1.55.0 and Chromium:

```bash
.venv-ultrasound/bin/pip install playwright==1.55.0
.venv-ultrasound/bin/python -m playwright install chromium
# With the live simulation running, on a desktop display:
DISPLAY=:1 .venv-ultrasound/bin/python scripts/check_live_brain.py
# Separately, reset and exercise repeated automatic scans for 100 seconds:
.venv-ultrasound/bin/python scripts/check_live_stability.py
```

It checks actual pressing, pause, reset, scanning and WebGL errors, and records
the real browser session under `outputs/live_gaussians_validation`.
For a stopped live run with tissue still deformed,
`.venv-ultrasound/bin/python scripts/validate_episode.py outputs/live_<timestamp> --live-final`
checks every tetrahedron and bonded interface in its final nodal snapshot.
That check covers the saved final state only, not every intermediate step.

## Patient brain, tumour, vessels and heterogeneous stiffness

A brain demo now uses **ReMIND-001's actual cerebrum, glioma and ventricle
segmentations**, downloaded from [TCIA ReMIND](https://www.cancerimagingarchive.net/collection/remind/)
(DOI 10.7937/3RAG-D070, CC BY 4.0). The importer respects each DICOM SEG's
orientation, spacing and per-frame physical positions, puts the structures into
the same patient coordinate frame, then applies one common metric rigid
transform for probe access. Surface meshes are exported as PLY, in metres.
Raw downloads, source series identifiers, licence and SHA-256 hashes are retained.

```bash
# From this project directory, after scripts/install.sh:
bash scripts/run_brain_demo.sh outputs/new_brain_demo

# Prepare or adjust the anatomy independently:
.venv-ultrasound/bin/python scripts/prepare_brain.py \
  --spacing-mm 5 --brain-youngs 3000 --variation .3 --tumor-ratio .7

# View the brain and acoustic preview in Isaac:
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/run_isaac.py \
  --anatomy data/remind-001/anatomy.npz --probe-motion sweep \
  --ultrasound-preview --seconds 2 --fps 2 --scatterers 40000 \
  --output outputs/brain_interactive

# Download true segmented arteries AND veins from a separate TopBrain subject:
.venv-ultrasound/bin/python scripts/prepare_vessels.py
```

The brain's stiffness varies spatially across three bonded FEM regions with
Young's moduli **2.1, 3.0 and 3.9 kPa**. The glioma has a separately adjustable
modulus (2.1 kPa by default). These are illustrative model parameters, **not
measurements from the patient**. The smooth generating field is discretised into
material bands; these bands are not grey/white matter segmentations. Tumours
are not assumed always to be stiffer. Omni PhysX currently does not implement
multi-material `UsdSubset` binding on a single deformable, so each material
region is a nonoverlapping volume with vertex attachments along all shared
interfaces. This changes the physics, not merely the display colour.

`--probe-motion sweep` presses through 4 mm, moves in both lateral directions,
turns up to 15 degrees about its contact normal, then lifts. It follows a
prescribed path, not force control. Each B-mode image uses the recorded rigid
probe pose and deformed material scatterers. The brain preset images to 50 mm
depth, samples scatterers in the local swept imaging region, and uses a brighter
glioma scatterer distribution; this acoustic contrast is an assumption.

The prototype uses a 5 mm Cartesian tetrahedral volume mesh, a fixed base,
and finer segmentation skins moved by fixed inverse-distance interpolation of
the FEM displacement. Ventricles are excluded from the solid volume and shown
in blue; **CSF fluid dynamics, skull and meninges are not modelled**. The purple
object is the patient's glioma. The cutaway removes display triangles only.
Elasticity damping is not a calibrated viscoelastic brain constitutive model.
The current linear array and homogeneous single-scattering acoustics remain
the same approximations described below; this is not a clinically validated
neurosurgical ultrasound simulator.

[TopBrain](https://zenodo.org/records/16878417) supplies actual labelled cerebral
arteries and veins. `prepare_vessels.py` retrieves one segmentation using HTTP
range requests, avoiding a 2 GB full archive download, and exports `arteries.ply`
and `veins.ply` under `data/topbrain-001`. The veins include the vein of Galen,
straight sinus, internal cerebral veins, basal veins and superior sagittal sinus.
TopBrain permits non-commercial use with attribution; commercial use requires
data-owner permission (see the downloaded `raw/License.txt`). **These vessels
are from a different subject and are kept separate from the ReMIND simulation.**
They require an explicit, assessed anatomical registration before forming a
composite, or vascular segmentation of the same patient's suitable imaging for
a patient-specific model. The current brain video contains no vascular model,
blood-flow simulation or Doppler ultrasound.

Prepared assets: `data/remind-001/{cerebrum,tumor,ventricles}.ply`, `anatomy.npz`,
`provenance.json`, and `data/topbrain-001/{arteries,veins}.ply` with its own
provenance. `outputs/brain_sweep` is the validated first brain episode. Its
physics validation recorded 4.78 mm peak tissue motion, no inverted tetrahedra,
under 0.01 mm interface separation, a 4 mm pressing stroke and roughly 12 mm
lateral probe travel. An otherwise identical uniform-stiffness control changed
nodal positions by up to 1.07 mm (maximum frame RMS difference 0.178 mm), confirming
that the material map affects physical motion. See `heterogeneity_validation.json`
in the episode.

## Tumour scan and gentle pressing

For a wider tumour-targeting scan with a pause and an extra 2 mm press:

```bash
bash scripts/run_tumor_scan.sh outputs/new_tumor_scan
```

This uses `--probe-motion scan --seconds 6`, follows a conservative envelope of
the FEM surface, and crosses 64 mm in elevation so the slice enters and leaves
the lesion. The extra press increases the commanded travel from 3 to 5 mm,
then eases back before scanning onward and lifting. The 3D cyan rectangle shows
the recorded acoustic plane. The video overlays the intersection of that plane
with the deformed tumour surface in yellow; **this is simulation ground truth,
not a prediction from ultrasound**. Raw B-mode PNG/RF/IQ files remain separate.
One shared display gain is derived from the entire scan, avoiding saturation
when the initial frame contains little tissue. The scan's acoustic ROI is
expanded, and scatterers farther than 8 mm from the elevation plane are omitted
as a rendering approximation. The downloaded vessels remain separate.

For a more conspicuous tumour in the simulated ultrasound, use
`--tumor-backscatter 6` on the Isaac runner. This sets the tumour scatterer echo
amplitude to six times that of brain, compared with the original 1.5 times
(a 12 dB increase). It is an illustrative visibility setting, not a measured
tumour density. The scan helper defaults to this stronger contrast; set
`TUMOR_BACKSCATTER=1.5` to retain the original contrast. Mechanical density and
stiffness are separate parameters.

Existing mechanics can be retained while regenerating the acoustic observation:

```bash
.venv-ultrasound/bin/python scripts/set_tumor_contrast.py outputs/tumor_scan \
  --output outputs/tumor_scan_contrast --tumor-backscatter 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-ultrasound/bin/python \
  scripts/render_acoustic_sequence.py outputs/tumor_scan_contrast --workers 12
.venv-ultrasound/bin/python scripts/make_video.py outputs/tumor_scan_contrast \
  --no-tumor-outline
.venv-ultrasound/bin/python scripts/compare_tumor_contrast.py \
  outputs/tumor_scan outputs/tumor_scan_contrast
```

The contrast variant changes only tumour scatterer coefficients; reference
geometry, other scatterers and all recorded mechanical states are preserved.
RF/IQ are recalculated before B-mode display, rather than painting the tumour
into the image. The comparison uses the same display gain and recorded pose in
both panels, without a reference outline.

## Original block phantom

Isaac Sim 6.2 runs a GPU volumetric deformable phantom under a contacting rigid
compression tool. A separate PyMUST/SIMUS process generates RF channel data from
3D material-embedded scatterers, then reconstructs ultrasound B-mode images.

The phantom is a 40×20×40 mm block. A matrix with a cavity and a conforming
inclusion have separate elastic materials (15 and 60 kPa), joined by vertex
attachments at their shared boundary. The base is anchored. A top tool indents
the block by up to 3 mm, holds and releases. A fixed side-facing probe avoids
requiring tracked probe motion.

## Run

Two environments isolate Isaac's pinned packages from the acoustic packages:

```bash
# Install with a supported NVIDIA driver and Python 3.12 + venv.
bash scripts/install.sh
export OMNI_KIT_ACCEPT_EULA=YES

# Full mechanics -> acoustic image sequence (offline acoustic rendering).
bash scripts/run_demo.sh outputs/new_demo

# Isaac viewport: omit --headless.
.venv/bin/python scripts/run_isaac.py --ultrasound-preview \
  --seconds 2 --fps 2 --scatterers 20000 --output outputs/interactive

# Render an existing episode, retaining RF and complex IQ:
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-ultrasound/bin/python \
  -m deform_sim.ultrasound outputs/interactive

# Small acoustic integration check, not the image-quality preset:
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-ultrasound/bin/python \
  -m deform_sim.ultrasound outputs/interactive --quick --frames 2
```

Output directories must be new to prevent silently replacing experiments.
Each run writes `scene.usda`, `reference.npz`, `metadata.json`, state snapshots,
and a completion record. Ultrasound rendering adds PNG frames, an animated GIF,
raw RF channels, complex beamformed IQ, commanded loading and the image grid. The saved USD scene
can also be opened in Isaac Sim. With `--ultrasound-preview`, a separate acoustic
worker fills a B-mode window inside Isaac as frames become available. Acoustic
simulation is asynchronous and slower than the physics loop, not real time.
After physics finishes, the viewer replays the mechanical state matching each
completed ultrasound frame. Preview uses 64 elements and one angle; pass
`--ultrasound-quality standard` for 128 elements and three angles.
The small demo uses 20,000 scatterers and five snapshots; measured acoustic
preview time here is approximately 25–30 seconds per frame. The simulator's
default export uses 120,000 scatterers at 20 Hz, which takes substantially longer
to render. Use `--output` on the acoustic CLI to keep multiple reconstructions.
GUI mode selects the Xorg display GPU automatically; override with `--gpu N`.

For a moving ultrasound head that itself compresses the phantom, add
`--probe-motion press`. This mounts the 40 mm linear head above the tissue,
parks and hides the separate tool, and moves the probe down through a 4 mm
stroke before lifting it back up. The contact face indents the tissue by about
3 mm. A nominal 3 mm acoustic standoff separates the aperture from the contact
face; the observation renderer transforms the persistent scatterers into the
saved probe frame for every image. This uses simulated pose ground truth for
image formation, not an external probe tracker.

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/run_isaac.py \
  --probe-motion press --ultrasound-preview --seconds 2 --fps 2 \
  --scatterers 20000 --output outputs/probe_session
```

## Ultrasound formation

The default virtual linear array has 128 elements, 0.3 mm pitch, 5 MHz centre
frequency, finite elevation aperture and focus, 70% bandwidth, and three plane
wave transmit angles. SIMUS simulates echoes, element directivity, pulse response
and frequency-dependent homogeneous attenuation. RF is demodulated to IQ,
delay-and-sum beamformed, coherently compounded and log-compressed over 60 dB.
Display gain is fixed from the first frame. Scatterer amplitudes and tetrahedral
barycentric coordinates are sampled once; their world positions follow the
actual PhysX simulation nodes at every frame. The inclusion is hypoechoic.

These are acoustic simulations, not rendered grayscale anatomy with added noise.
Realistic appearance still needs calibration to your probe and phantom.
SIMUS assumes a homogeneous propagation medium: it does not establish accurate
refraction, multiple scattering, reverberation or heterogeneous nodule shadowing.
A full-wave backend such as k-Wave is appropriate for those validation cases.
The mechanics currently uses elasticity damping; it does not implement a
validated tissue viscoelastic constitutive law.

The imaging frame is fixed: lateral = world x, elevation = world z − 20 mm,
depth = world y + 15 mm. There is a nominal 5 mm homogeneous coupling path.
The compression tool is on the top (+z); the probe is on the side (−y).
The GPU backend does not implement rigid kinematic-target tensor writes. The
runner applies small rigid-pose increments at 240 Hz before each contact solve;
this is displacement-controlled loading, not force control.

## Evaluation and checks

A synchronized demonstration video is available at
`outputs/video_episode/video/simulation_ultrasound.mp4`, with an animated
`preview.gif`. It pairs 41 Isaac-rendered states with 41 independently simulated
ultrasound frames by snapshot ID and timestamp. The clip plays at half speed and
repeats twice. The 3D cutaway reveals the inclusion; mechanics are simulated in
the intact volume.

`outputs/moving_probe/video/simulation_ultrasound.mp4` demonstrates the moving
ultrasound head pressing and lifting. Its 41 acoustic frames use the corresponding
saved aperture position and orientation; the video composer checks that both
poses and timestamps agree with the mechanical snapshots. This episode reached
3.36 mm peak tissue displacement during a 4 mm probe stroke, with the separate
compression tool parked and no inverted tetrahedra in the recorded states.

To generate another video:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/run_isaac.py \
  --headless --seconds 2 --fps 20 --scatterers 20000 --output outputs/new_video
.venv-ultrasound/bin/python scripts/validate_episode.py outputs/new_video
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-ultrasound/bin/python \
  scripts/render_acoustic_sequence.py outputs/new_video --workers 6
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/capture_replay.py outputs/new_video
.venv-ultrasound/bin/python scripts/make_video.py outputs/new_video
```

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-ultrasound/bin/python \
  -m unittest discover -s tests -v
```

Checks cover conforming inclusion attachment positions, positive tetrahedra and
total phantom volume, affine scatterer transport, acoustic point-target
localization, fixed display gain, and draining snapshots during asynchronous
rendering. The Isaac runner fails if a full loading
episode produces no recorded deformation.

`state/*.npz` and material metadata are privileged ground truth, not model
inputs. The ultrasound PNG/RF/IQ sequences are observations. Hidden markers
and full volume node positions support evaluation outside the imaging plane.
Specimen-level benchmark splits, force measurement, mechanical calibration,
real-image validation and a recurrent world model are still to be built.

The verified example episode is in `outputs/isaac_demo`, with a five-frame
ultrasound sequence, a separate standard-quality reconstruction and an actual
Isaac window capture (`isaac_window.png`). The GPU episode reached 3.51 mm peak
displacement; all sampled tetrahedra retained positive volume, and inclusion
interface separation stayed below 2 nm. These are numerical integration checks,
not validation against a physical specimen. Twelve automated checks pass, including
multi-region scatterer transport and motion of the detailed anatomical skin.
Earlier
prototype and integration-debug outputs are kept under `outputs/debug`.

## References

- [Isaac Sim Python installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)
- [PhysX volumetric deformables and attachments](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/deformable_bodies.html)
- [PyMUST / SIMUS](https://github.com/creatis-ULTIM/PyMUST)
- [k-Wave B-mode example](https://www.k-wave.org/documentation/example_us_bmode_linear_transducer.php)
