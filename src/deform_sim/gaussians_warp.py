"""GPU material transport, sharing the Isaac Warp runtime (no extra CUDA stack)."""
import numpy as np
import warp as wp


@wp.kernel
def _volume_jacobians(nodes: wp.array(dtype=wp.vec3), ids: wp.array(dtype=wp.vec4i),
                      inverse: wp.array(dtype=wp.mat33), result: wp.array(dtype=float)):
    i = wp.tid()
    index = ids[i]
    a = nodes[index[0]]
    b, c, d = nodes[index[1]]-a, nodes[index[2]]-a, nodes[index[3]]-a
    edges = wp.mat33(b[0], c[0], d[0], b[1], c[1], d[1], b[2], c[2], d[2])
    result[i] = wp.determinant(edges*inverse[i])


@wp.kernel
def _advect(nodes: wp.array(dtype=wp.vec3), rest: wp.array(dtype=wp.vec3),
            ids: wp.array(dtype=wp.vec4i), weights: wp.array(dtype=wp.vec4),
            inverse: wp.array(dtype=wp.mat33), covariance: wp.array(dtype=wp.mat33),
            youngs: wp.array(dtype=float), soft_count: int, rotation: wp.mat33,
            translation: wp.vec3, result: wp.array2d(dtype=float)):
    i = wp.tid()
    center = wp.vec3(0.0)
    cov = wp.mat33(0.0)
    displacement = float(0.0)
    von_mises = float(0.0)
    jacobian = float(1.0)
    if i < soft_count:
        index = ids[i]
        a, b, c, d = nodes[index[0]], nodes[index[1]], nodes[index[2]], nodes[index[3]]
        e1, e2, e3 = b-a, c-a, d-a
        edges = wp.mat33(e1[0], e2[0], e3[0], e1[1], e2[1], e3[1], e1[2], e2[2], e3[2])
        f = edges*inverse[i]
        w = weights[i]
        center = w[0]*a+w[1]*b+w[2]*c+w[3]*d
        cov = f*covariance[i]*wp.transpose(f)
        displacement = wp.length(center-rest[i])
        jacobian = wp.determinant(f)
        safe_j = wp.max(jacobian, 1.0e-6)
        mu = youngs[i]/2.9
        lam = youngs[i]*0.45/(1.45*0.1)
        identity = wp.identity(n=3, dtype=float)
        stress = (mu/safe_j)*(f*wp.transpose(f)-identity)+(lam*wp.log(safe_j)/safe_j)*identity
        mean = wp.trace(stress)/3.0
        total = float(0.0)
        for row in range(3):
            for col in range(3):
                value = stress[row, col]
                if row == col:
                    value = value-mean
                total = total+value*value
        von_mises = wp.sqrt(1.5*total)
    else:
        center = rotation*rest[i]+translation
        cov = rotation*covariance[i]*wp.transpose(rotation)
    result[i, 0] = center[0]
    result[i, 1] = center[1]
    result[i, 2] = center[2]
    result[i, 3] = cov[0, 0]
    result[i, 4] = cov[0, 1]
    result[i, 5] = cov[0, 2]
    result[i, 6] = cov[1, 1]
    result[i, 7] = cov[1, 2]
    result[i, 8] = cov[2, 2]
    result[i, 9] = displacement
    result[i, 10] = von_mises
    result[i, 11] = jacobian


class WarpGaussianTransport:
    def __init__(self, model, device, bodies=None):
        self.device, self.model = device, model
        self.count, self.soft_count = len(model['means']), int(model['soft_count'])
        self.arrays = [wp.array(model[key], dtype=dtype, device=device) for key, dtype in
            [('means', wp.vec3), ('nodes', wp.vec4i), ('weights', wp.vec4),
             ('inverse_rest_edges', wp.mat33), ('covariance', wp.mat33), ('youngs', wp.float32)]]
        self.result = wp.empty((self.count, 12), dtype=wp.float32, device=device)
        self.volumes = None
        if bodies is not None:
            offsets = np.r_[0, np.cumsum([len(b.points) for b in bodies])]
            tets = np.concatenate([b.tetrahedra+offsets[i] for i, b in enumerate(bodies)])
            vertices = np.concatenate([b.points for b in bodies])[tets]
            inverse = np.linalg.inv((vertices[:, 1:]-vertices[:, :1]).transpose(0, 2, 1))
            self.volumes = (wp.array(tets, dtype=wp.vec4i, device=device),
                            wp.array(inverse, dtype=wp.mat33, device=device),
                            wp.empty(len(tets), dtype=wp.float32, device=device))

    def pack(self, positions, probe_pose, header):
        nodes = wp.array(np.concatenate(positions), dtype=wp.vec3, device=self.device)
        x, y, z, w = map(float, probe_pose[3:])
        rotation = wp.quat_to_matrix(wp.quat(x, y, z, w))
        wp.launch(_advect, dim=self.count, inputs=[nodes, *self.arrays, self.soft_count,
            rotation, wp.vec3(*map(float, probe_pose[:3])), self.result], device=self.device)
        if self.volumes is not None:
            wp.launch(_volume_jacobians, dim=len(self.volumes[0]), inputs=[nodes, *self.volumes], device=self.device)
        values = self.result.numpy()
        minimum_j = float((self.volumes[2].numpy() if self.volumes is not None else values[:self.soft_count, 11]).min())
        displacement, stress = float(values[:, 9].max()), float(values[:, 10].max())
        h = np.zeros(16, np.float32)
        h[:len(header)] = header
        h[10:13] = minimum_j, displacement, stress
        h[14] = minimum_j <= 0 or not np.isfinite(displacement)
        return h.tobytes()+values[:, :11].copy().tobytes(), dict(min_jacobian=minimum_j,
            jacobian_scope='all FEM tetrahedra' if self.volumes is not None else 'Gaussian binding tetrahedra',
            max_displacement_mm=displacement*1000, max_constitutive_stress_pa=stress)
