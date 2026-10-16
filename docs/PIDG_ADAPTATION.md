# Material Gaussians driven by an anatomical prior

The requested paper is Hong et al., *Physics-Informed Deformable Gaussian
Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field*,
[arXiv:2511.06299](https://arxiv.org/abs/2511.06299),
[AAAI publication](https://doi.org/10.1609/aaai.v40i6.42474).
Its [official implementation](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting)
is an optical dynamic-scene reconstruction system. We reviewed its method and
source; this project's viewer and transport implementation are independent.

## What this implementation does

The user permitted an MRI/CT or mesh prior. We use the existing aligned
ReMIND-001 segmented cerebrum, glioma and ventricles. Area-weighted samples of
the fine triangle surfaces initialise anisotropic Gaussian centres, tangential
scales, normal thickness, colours and opacity. Colours are illustrative anatomy
colours, not learned MRI intensities or optical appearance. These are surface
Gaussians, not a dense reconstructed tissue volume.

Each soft Gaussian has fixed barycentric material coordinates in a coarse FEM
tetrahedron. For current vertices `x` and rest vertices `X`, the deformation
gradient is `F = [x1-x0, x2-x0, x3-x0] [X1-X0, X2-X0, X3-X0]^-1`.
The centre follows the barycentric combination of current vertices, and the
covariance follows `Sigma = F Sigma_0 F^T`. This transports shape and orientation
under affine compression, shear and rotation. Probe and plane Gaussians follow
the rigid probe pose. Warp evaluates transport on the same GPU as PhysX.

Isaac's GPU volumetric deformables supply motion, contact, anchoring and bonded
spatially heterogeneous elastic regions. Browser controls change the contacting
probe during execution. The WebSocket sends only the latest state, allowing the
viewer to drop stale frames without blocking the solver. The WebGL2 renderer
projects covariances through the camera Jacobian, draws elliptical Gaussian
footprints, sorts by centre depth and composites alpha from back to front.

The stress colouring evaluates a compressible Neo-Hookean Cauchy law with the
assumed regional modulus and Poisson ratio 0.45, then displays von Mises stress.
It is **an analytic diagnostic**, not PhysX stress readback, measured tissue
stress, or a learned constitutive law. A second GPU kernel checks volume ratios
of all FEM tetrahedra at each streamed update. The live loop pauses and the
viewer reports an invalid state if any has nonpositive determinant. This does
not check every intermediate substep or establish convergence of the mechanics.

## Relationship to the paper

| Component | Paper | This implementation |
|---|---|---|
| Geometry prior | Optimised dynamic optical Gaussian scene | Segmented patient surface mesh |
| Motion | Learned deformation / material-field velocity | Live Isaac FEM nodal motion |
| Material representation | Time-evolving velocity and stress field | Fixed heterogeneous modulus prior and analytic diagnostic |
| Physics | Momentum-balance residual in training | Explicit solver contact and deformation |
| Appearance fitting | Multi-view/temporal optical losses and motion supervision | Illustrative fixed anatomy colours |
| Ultrasound reconstruction | Not established by the paper's optical experiments | Not implemented; existing offline RF renderer remains separate |

The learned time-dependent velocity/stress network, camera/optical-flow training,
physics-residual optimisation and constitutive discovery have not been added.
Those require a training dataset, supervision and held-out evaluation; replacing
optical losses with ultrasound losses is a separate research task. This prototype
provides a live forward model and a material-bound Gaussian representation on
which that work can build. It does not claim to have learned the patient's
mechanics or recovered a tumour from live ultrasound.

## Numerical and anatomical limits

The 5 mm FEM boundary does not coincide with every fine segmentation triangle.
The nearest suitable tetrahedron is selected by barycentric-violation penalty;
surface samples outside it retain extrapolating coordinates. The preparation
JSON reports that fraction and minimum weight. For the current prior about
47% of samples extrapolate, with some weights as low as -4.69. This preserves
the fine rest surface and exact affine motion, but can amplify local nonlinear
motion outside the coarse boundary. It needs mesh-conforming refinement before
quantitative use near the boundary. Gaussians do not independently collide.

Stiffness values are illustrative, not patient measurements. Ventricles are
displayed as moving anatomical surfaces without CSF flow. There is no skull,
meninges, vascular flow, vessel registration, tissue damage or force-controlled
probe. Gaussian transparency and cutaways are display operations. Increasing
motion magnification does not change the solver or acoustic data.

Automatic scanning uses 3 mm maximum commanded travel and a 12-second pass.
Terrain following evaluates the actual rate-limited probe position, avoiding
overcompression when the head lags a target on a slope. Manual vertical travel
is limited to 8 mm/s downward, with faster lifting to follow rising terrain.

Live contact uses a configurable smaller substep count than the 240 Hz offline
preset. Report the measured solver update rate, browser rate, simulated-time /
wall-time ratio and determinant checks separately. Video frame rate alone is
not evidence of real-time physics. `scripts/check_live_brain.py` exercises the
running solver through the same browser controls used interactively.

## Measured prototype check, 2026-10-16

On this workstation (RTX 2080 Ti for Isaac and an RTX 2080 Ti driving the browser),
the browser integration check measured a median 29.8 solver/Gaussian updates/s
and 35.2 viewer frames/s while recording a 1440 x 1000 browser video. It exercised
5 mm commanded pressing, automatic scanning, pause and reset, with 45,192
Gaussians and 50,838 FEM tetrahedra. Peak displayed surface motion at the pressing
check was 4.61 mm. Reset returned surface displacement below 0.001 mm, and no
WebGL errors were reported. Recording overhead affects the viewer rate.

A separate saved final snapshot under pressure had 4.23 mm maximum FEM nodal
motion, minimum volume ratio 0.713 over **all** tetrahedra, and 3.48 micrometres
maximum bonded-interface separation. This is a final-state check, not a full
trajectory positivity proof. Fifteen CPU/acoustic checks and the additional
Warp transport comparison passed across the two project environments.
These establish software behaviour, not agreement with a physical brain.

Local evidence is saved under `outputs/live_gaussians_final` (browser checks,
screenshots and MP4), and `outputs/live_20261016_120014/live_validation.json`
(full final-state check). Generated data and videos are not tracked in Git.

The subsequent repeated-scan check used the corrected terrain following,
full-mesh GPU volume checks, direct simulation-node reads and absolute frame
deadlines. It ran for 100.49 wall seconds with a simulated-time/wall-time ratio
of 0.974; the median instantaneous solver update rate was 30.6 Hz. The lowest
observed full-mesh volume ratio in its 1 Hz report was 0.822, automatic travel
stayed at or below 3 mm, and no invalid-state pause occurred. The live renderer
was also exercised during the run: its separate 60-second observer reported
34.0 median frames/s and no WebGL errors. Evidence is under
`outputs/live_gaussians_native/{repeated_scan_final,viewer_observer}.json`.
These rates distinguish solver updates, displayed frames and actual advancement
of the simulation clock.
