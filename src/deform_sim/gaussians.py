"""Material-bound anisotropic Gaussians advected by the live FEM deformation.

This is an explicit constitutive-prior adaptation of PIDG, not its trained
monocular reconstruction network. Covariance follows F Sigma_0 F^T.
"""
import numpy as np


def deformation_gradient(vertices, inverse_rest_edges):
    edges = (vertices[:, 1:]-vertices[:, :1]).transpose(0, 2, 1)
    return edges@inverse_rest_edges


def neo_hookean_diagnostics(gradient, youngs, poisson=.45):
    """Compressible Neo-Hookean Cauchy/von Mises stress, a diagnostic in Pa.

    It is an analytic constitutive prior, not a readback of PhysX solver stress.
    """
    jacobian = np.linalg.det(gradient)
    safe_j = np.maximum(jacobian, 1e-6)
    mu = youngs/(2*(1+poisson))
    lam = youngs*poisson/((1+poisson)*(1-2*poisson))
    left_cauchy_green = gradient@gradient.transpose(0, 2, 1)
    stress = mu[:, None, None]/safe_j[:, None, None]*(left_cauchy_green-np.eye(3))
    stress += (lam*np.log(safe_j)/safe_j)[:, None, None]*np.eye(3)
    deviator = stress-np.trace(stress, axis1=1, axis2=2)[:, None, None]/3*np.eye(3)
    von_mises = np.sqrt(1.5*np.sum(deviator**2, axis=(1, 2)))
    return jacobian, von_mises


def advect_gaussians(model, positions, probe_pose):
    """Return live centers, full covariances, displacement, stress, minimum J."""
    n = int(model['soft_count'])
    nodes = np.concatenate(positions)
    vertices = nodes[model['nodes'][:n]]
    gradient = deformation_gradient(vertices, model['inverse_rest_edges'][:n])
    centers = np.empty_like(model['means'])
    centers[:n] = np.einsum('ni,nij->nj', model['weights'][:n], vertices)
    covariance = np.empty_like(model['covariance'])
    covariance[:n] = gradient@model['covariance'][:n]@gradient.transpose(0, 2, 1)
    displacement = np.zeros(len(centers), np.float32)
    displacement[:n] = np.linalg.norm(centers[:n]-model['means'][:n], axis=1)
    j, stress = neo_hookean_diagnostics(gradient, model['youngs'][:n])
    von_mises = np.zeros(len(centers), np.float32)
    von_mises[:n] = stress
    # Rigid probe and imaging plane have their own canonical local coordinates.
    x, y, z, w = probe_pose[3:]
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    centers[n:] = model['means'][n:]@rotation.T+probe_pose[:3]
    covariance[n:] = rotation@model['covariance'][n:]@rotation.T
    return centers, covariance, displacement, von_mises, float(j.min())


def pack_gaussian_frame(model, positions, probe_pose, header):
    """Version 1: 16 float header then N x 11 floats; all SI units.

    Header 10:13 carries minimum J, peak displacement and stress; 13 is
    paused (supplied by the caller), 14 flags an invalid tissue state.
    """
    centers, cov, displacement, stress, minimum_j = advect_gaussians(model, positions, probe_pose)
    data = np.empty((len(centers), 11), np.float32)
    data[:, :3] = centers
    data[:, 3:9] = cov[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    data[:, 9], data[:, 10] = displacement, stress
    h = np.zeros(16, np.float32)
    h[:len(header)] = header
    h[10:13] = minimum_j, displacement.max(), stress.max()
    h[14] = minimum_j <= 0 or not np.isfinite(displacement.max())
    return h.tobytes()+data.tobytes(), dict(min_jacobian=minimum_j,
        max_displacement_mm=float(displacement.max()*1000), max_constitutive_stress_pa=float(stress.max()))
