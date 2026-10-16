"""Tracked B-mode reconstruction. No simulator deformation or labels are inputs."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class Observation:
    image: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    time: float
    lateral_min: float = -.02
    lateral_max: float = .02
    depth_min: float = .002
    depth_max: float = .05

    def world_points(self, rows=None, cols=None):
        if rows is None:
            rows, cols = np.indices(self.image.shape)
        local = np.stack((self.lateral_min+cols/(self.image.shape[1]-1)*(self.lateral_max-self.lateral_min),
                          np.zeros_like(rows),
                          self.depth_min+rows/(self.image.shape[0]-1)*(self.depth_max-self.depth_min)), axis=-1)
        return local@self.rotation.T+self.position

    def sample_world(self, world):
        local = (world-self.position)@self.rotation
        col = (local[..., 0]-self.lateral_min)/(self.lateral_max-self.lateral_min)*(self.image.shape[1]-1)
        row = (local[..., 2]-self.depth_min)/(self.depth_max-self.depth_min)*(self.image.shape[0]-1)
        values = map_coordinates(self.image, [row.ravel(), col.ravel()], order=1, mode='constant', cval=0).reshape(row.shape)
        inside = (row >= 0)&(row <= self.image.shape[0]-1)&(col >= 0)&(col <= self.image.shape[1]-1)
        return values, local[..., 1], inside


class TrackedVolume:
    def __init__(self, lower, upper, spacing=.001, elevation_sigma=.0015):
        self.origin = np.asarray(lower, dtype=np.float32)
        self.spacing, self.elevation_sigma = spacing, elevation_sigma
        self.shape = tuple((np.ceil((np.asarray(upper)-self.origin)/spacing).astype(int)+1).tolist())
        self.points = self.origin+np.indices(self.shape, dtype=np.float32).reshape(3, -1).T*spacing
        self.total = np.zeros(self.shape, np.float32)
        self.weight = np.zeros(self.shape, np.float32)
        self.frames = 0

    @classmethod
    def from_observations(cls, observations, **kwargs):
        corners = np.concatenate([o.world_points(np.array([0, 0, o.image.shape[0]-1, o.image.shape[0]-1]),
            np.array([0, o.image.shape[1]-1, 0, o.image.shape[1]-1])) for o in observations])
        return cls(corners.min(axis=0)-.005, corners.max(axis=0)+.005, **kwargs)

    def add(self, observation):
        values, distance, inside = observation.sample_world(self.points)
        valid = inside&(np.abs(distance) <= 3*self.elevation_sigma)
        weights = np.exp(-.5*(distance/self.elevation_sigma)**2)*valid
        self.total.ravel()[:] += (values*weights).astype(np.float32)
        self.weight.ravel()[:] += weights.astype(np.float32)
        self.frames += 1

    @property
    def intensity(self):
        return self.total/np.maximum(self.weight, 1e-8)

    def sample(self, points):
        coords = ((np.asarray(points)-self.origin)/self.spacing).reshape(-1, 3).T
        total = map_coordinates(self.total, coords, order=1, mode='constant', cval=0)
        weight = map_coordinates(self.weight, coords, order=1, mode='constant', cval=0)
        return (total/np.maximum(weight, 1e-8)).reshape(points.shape[:-1]), weight.reshape(points.shape[:-1])

    def splats(self, limit=35000, minimum=.24):
        value, weight = self.intensity.ravel(), self.weight.ravel()
        ids = np.flatnonzero((value >= minimum)&(weight >= .08))
        if len(ids) > limit:
            ids = ids[np.linspace(0, len(ids)-1, limit).astype(int)]
        # x,y,z, grey intensity, opacity; no semantic lesion label is inferred.
        return np.column_stack((self.points[ids], value[ids], np.minimum(1, weight[ids]))).astype(np.float32)


def estimate_inplane_motion(anchor, current):
    """Pose-compensated image flow; reject out-of-plane pairs explicitly."""
    from scipy.ndimage import gaussian_filter, binary_erosion
    from skimage.registration import optical_flow_tvl1
    normal_distance = abs(float((current.position-anchor.position)@anchor.rotation[:, 1]))
    angle = float(np.dot(anchor.rotation[:, 1], current.rotation[:, 1]))
    if normal_distance > .0005 or angle < np.cos(np.deg2rad(1)):
        return None
    world = anchor.world_points()
    aligned, _, inside = current.sample_world(world)
    reference = gaussian_filter(anchor.image, .7)
    moving = gaussian_filter(aligned, .7)
    v, u = optical_flow_tvl1(reference, moving, attachment=15, tightness=.3, num_warp=5, num_iter=10)
    rv, ru = optical_flow_tvl1(moving, reference, attachment=15, tightness=.3, num_warp=5, num_iter=10)
    rows, cols = np.indices(anchor.image.shape)
    backward_v = map_coordinates(rv, [rows+v, cols+u], order=1, mode='nearest')
    backward_u = map_coordinates(ru, [rows+v, cols+u], order=1, mode='nearest')
    consistency = np.sqrt((v+backward_v)**2+(u+backward_u)**2)
    valid = binary_erosion(inside&(reference>.20)&(aligned>.20), iterations=4)
    confidence = np.exp(-.5*(consistency/1.5)**2)*valid
    dx = (anchor.lateral_max-anchor.lateral_min)/(anchor.image.shape[1]-1)
    dz = (anchor.depth_max-anchor.depth_min)/(anchor.image.shape[0]-1)
    displacement = (u[..., None]*dx*anchor.rotation[:, 0]+v[..., None]*dz*anchor.rotation[:, 2]).astype(np.float32)
    confidence *= np.linalg.norm(displacement, axis=2) < .004
    return dict(displacement=displacement, confidence=confidence.astype(np.float32),
                flow_pixels=np.stack((u, v), axis=-1), aligned_image=aligned.astype(np.float32))


def advance_material_tracks(previous, current, initial_world, inferred_displacement, cumulative, reference=None):
    """Track adjacent image pairs at prior-predicted material locations.

    Adjacent-pair tracking limits ultrasound speckle decorrelation compared
    with matching a strongly deformed image directly to the initial frame.
    """
    if reference is not None and current.time-reference.time > 1.5:
        aligned, distance, inside = current.sample_world(reference.world_points())
        visible = inside&(reference.image>.2)&(aligned>.2)
        if visible.sum()>200 and np.abs(distance[visible]).max()<.0005:
            a, b = reference.image[visible], aligned[visible]
            correlation = np.corrcoef(a, b)[0, 1]
            if correlation > .94 and np.sqrt(np.mean((a-b)**2)) < .04:
                # Image-based return to the reference appearance bounds
                # accumulated speckle-tracking drift without loading commands.
                return np.zeros_like(cumulative), np.ones(len(cumulative), np.float32)
    motion = estimate_inplane_motion(previous, current)
    if motion is None:
        return None
    local = (initial_world+inferred_displacement-previous.position)@previous.rotation
    cols = (local[:, 0]-previous.lateral_min)/(previous.lateral_max-previous.lateral_min)*(previous.image.shape[1]-1)
    rows = (local[:, 2]-previous.depth_min)/(previous.depth_max-previous.depth_min)*(previous.image.shape[0]-1)
    confidence = map_coordinates(motion['confidence'], [rows, cols], order=1, mode='constant', cval=0)
    incremental = np.column_stack([map_coordinates(motion['displacement'][..., j], [rows, cols], order=1,
        mode='constant', cval=0) for j in range(3)])
    target = cumulative+incremental*(confidence >= .35)[:, None]
    valid = np.linalg.norm(target, axis=1) <= .005
    confidence *= valid
    target[~valid] = cumulative[~valid]
    return target, confidence
