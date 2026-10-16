"""Infer incremental 3D motion from image flow and an initial elastic mesh prior.

No deformed simulator vertices, loading commands or reference tumour labels are
used as measurements. Out-of-plane displacement is a prior-based extrapolation.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator, splu
from scipy.linalg import solve as dense_solve
from .phantom import load_bodies


class ElasticInverse:
    def __init__(self, prior, poisson=.45, data_weight=5000.):
        self.bodies = load_bodies(prior)
        all_ids = np.concatenate([b.global_ids for b in self.bodies])
        _, first, mapping = np.unique(all_ids, return_index=True, return_inverse=True)
        self.points = np.concatenate([b.points for b in self.bodies])[first]
        self.mapping = mapping
        offsets = np.r_[0, np.cumsum([len(b.points) for b in self.bodies])]
        self.body_maps = [mapping[offsets[i]:offsets[i+1]] for i in range(len(self.bodies))]
        self.tets = np.concatenate([self.body_maps[i][b.tetrahedra] for i, b in enumerate(self.bodies)])
        e = np.repeat(prior['youngs_modulus'], [len(b.tetrahedra) for b in self.bodies])
        vertices = self.points[self.tets]
        edges = (vertices[:, 1:]-vertices[:, :1]).transpose(0, 2, 1)
        self.inverse_edges = np.linalg.inv(edges)
        volume = np.abs(np.linalg.det(edges))/6
        gradient = np.concatenate((-self.inverse_edges.sum(axis=1)[:, None], self.inverse_edges), axis=1)
        mu = e/(2*(1+poisson)); lam = e*poisson/((1+poisson)*(1-2*poisson))
        blocks = (lam[:, None, None, None, None]*gradient[:, :, None, :, None]*gradient[:, None, :, None, :]
                  +mu[:, None, None, None, None]*gradient[:, :, None, None, :]*gradient[:, None, :, :, None])
        dot = np.einsum('nai,nbi->nab', gradient, gradient)
        blocks += mu[:, None, None, None, None]*dot[:, :, :, None, None]*np.eye(3)
        element = (volume[:, None, None, None, None]*blocks).transpose(0, 1, 3, 2, 4).reshape(-1, 12, 12)
        dofs = (self.tets[:, :, None]*3+np.arange(3)).reshape(-1, 12)
        rows = np.broadcast_to(dofs[:, :, None], element.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], element.shape).ravel()
        self.stiffness = sparse.coo_matrix((element.ravel(), (rows, cols)), shape=(len(self.points)*3,)*2).tocsr()
        base = np.isclose(self.points[:, 2], self.points[:, 2].min())
        self.free = np.flatnonzero(~np.repeat(base, 3))
        self.k = self.stiffness[self.free][:, self.free]
        self.k += sparse.eye(len(self.free))*max(float(self.k.diagonal().mean())*1e-8, 1e-8)
        self.tree = cKDTree(vertices.mean(axis=1))
        self.data_weight = data_weight
        self.cached = None
        self.factor = None
        self.schur = None

    def constraints(self, world, directions):
        _, candidates = self.tree.query(world, k=min(32, len(self.tets)))
        first = self.points[self.tets[candidates, 0]]
        tail = np.einsum('nkij,nkj->nki', self.inverse_edges[candidates], world[:, None]-first)
        weights = np.concatenate((1-tail.sum(axis=2, keepdims=True), tail), axis=2)
        penalty = np.minimum(weights, 0.)**2
        choice = penalty.sum(axis=2).argmin(axis=1)
        selected = candidates[np.arange(len(world)), choice]
        weights = weights[np.arange(len(world)), choice]
        inside = weights.min(axis=1) >= -.025
        selected, weights = selected[inside], weights[inside]
        dofs = self.tets[selected, :, None]*3+np.arange(3)
        values = weights[:, None, :, None]*directions[None, :, None, :]
        shape = (len(weights), len(directions), 4, 3)
        rows = np.broadcast_to(np.arange(len(weights)*len(directions)).reshape(len(weights), len(directions), 1, 1), shape)
        cols = np.broadcast_to(dofs[:, None], shape)
        c = sparse.coo_matrix((values.ravel(), (rows.ravel(), cols.ravel())), shape=(len(weights)*len(directions), len(self.points)*3)).tocsr()
        return c[:, self.free], inside

    def prepare(self, world, directions, confidence):
        c, inside = self.constraints(world, directions)
        weights = np.repeat(np.asarray(confidence)[inside], len(directions))
        self.point_sampling, self.point_inside = self.constraints(world, np.eye(3))
        if c.shape[0] <= 700:
            # Geometry/prior remain fixed. Precompute elastic compliance once;
            # changing image confidences then require only a small dense solve.
            if self.factor is None:
                self.factor = splu(self.k.tocsc())
            response = np.empty((len(self.free), c.shape[0]))
            for first in range(0, c.shape[0], 16):
                response[:, first:first+16] = self.factor.solve(c[first:first+16].T.toarray())
            projected = c@response
            self.schur = response, .5*(projected+projected.T)
            self.cached = c, inside, weights, None, None, np.zeros(len(self.free))
            return inside
        self.schur = None
        weighted = sparse.diags(weights)
        a = (self.k+self.data_weight*(c.T@weighted@c)).tocsr()
        try:
            import pyamg
            preconditioner = pyamg.smoothed_aggregation_solver(a, symmetry='symmetric').aspreconditioner()
        except ImportError:
            diagonal = a.diagonal()
            preconditioner = LinearOperator(a.shape, matvec=lambda x: x/diagonal)
        self.cached = c, inside, weights, a, preconditioner, np.zeros(len(self.free))
        return inside

    def point_displacements(self):
        """Current inferred displacement at the original image material points."""
        result = np.zeros((len(self.point_inside), 3))
        result[self.point_inside] = (self.point_sampling@self.cached[-1]).reshape(-1, 3)
        return result

    def solve(self, measured, confidence=None):
        c, inside, base_weights, base_a, preconditioner, previous = self.cached
        weights, a = base_weights, base_a
        if confidence is not None:
            channels = len(base_weights)//int(inside.sum())
            reliability = np.repeat(np.asarray(confidence)[inside], channels)
            reliability = np.where(reliability >= .35, reliability, 0.)
            weights = base_weights*reliability
            if self.schur is None:
                a = (self.k+self.data_weight*(c.T@sparse.diags(weights)@c)).tocsr()
        target = np.asarray(measured)[inside].ravel()
        if self.schur is not None:
            response, projected = self.schur
            active = np.flatnonzero(weights > 0)
            dual = np.zeros(len(weights))
            if len(active):
                system = projected[np.ix_(active, active)]+np.diag(1/(self.data_weight*weights[active]))
                dual[active] = dense_solve(system, target[active], assume_a='pos')
            solved = response@dual
        else:
            rhs = self.data_weight*(c.T@(weights*target))
            solved, flag = cg(a, rhs, x0=previous, M=preconditioner, rtol=1e-5, atol=1e-9, maxiter=800)
            if flag != 0:
                raise RuntimeError(f'elastic inverse did not converge (CG {flag})')
        def state(vector):
            displacement = np.zeros(len(self.points)*3)
            displacement[self.free] = vector
            displacement = displacement.reshape(-1, 3)
            positions = self.points+displacement
            edges = (positions[self.tets[:, 1:]]-positions[self.tets[:, :1]]).transpose(0, 2, 1)
            return displacement, positions, float(np.linalg.det(edges@self.inverse_edges).min())
        displacement, positions, minimum_j = state(solved)
        accepted = np.isfinite(displacement).all() and minimum_j >= .3 and np.linalg.norm(displacement, axis=1).max() <= .004
        if not accepted:
            solved = previous
            displacement, positions, minimum_j = state(solved)
        self.cached = c, inside, base_weights, base_a, preconditioner, solved
        residual = c@solved-target
        return [positions[index] for index in self.body_maps], dict(
            min_volume_jacobian=minimum_j, max_inferred_motion_mm=float(np.linalg.norm(displacement, axis=1).max()*1000),
            image_constraint_rms_mm=float(np.sqrt(np.sum(weights*residual**2)/max(weights.sum(), 1e-20))*1000),
            measurement_count=int(np.count_nonzero(weights)), solver_converged=True, accepted=bool(accepted),
            rejection_reason=None if accepted else 'Tracking exceeded the 4 mm incremental prior bound or volume-ratio limit; last valid estimate retained.',
            solver='precomputed elastic compliance + confidence-weighted Schur solve' if self.schur is not None else 'CG',
            interpretation='incremental image-driven displacement; unobserved components follow the elastic prior')
