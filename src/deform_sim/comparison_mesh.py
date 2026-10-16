"""Conventional triangle skins transported by actual Isaac FEM displacement.

Vertex clustering reduces display resolution only. The averaged sparse skin
embedding follows measured simulator nodes; no Gaussian representation is used.
Full-resolution tumour contours remain separate from this display mesh.
"""
import numpy as np
from scipy import sparse


def cluster_skin(points, faces, nodes, weights, node_count, spacing):
    _, labels = np.unique(np.floor(points/spacing).astype(np.int64), axis=0, return_inverse=True)
    count = int(labels.max())+1
    population = np.bincount(labels)
    rest = np.column_stack([np.bincount(labels, weights=points[:, i])/population for i in range(3)])
    mapped = labels[faces]
    mapped = mapped[(mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 2] != mapped[:, 0])]
    _, unique = np.unique(np.sort(mapped, axis=1), axis=0, return_index=True)
    mapped = mapped[np.sort(unique)].astype(np.uint32)
    skin = sparse.coo_matrix((weights.ravel()/np.repeat(population[labels], 4),
        (np.repeat(labels, 4), nodes.ravel())), shape=(count, node_count)).tocsr()
    return rest, mapped, skin


def box(lower, upper):
    vertices = np.array([[x, y, z] for x in [lower[0], upper[0]] for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]])
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    # Separate corners by face for hard-edged probe normals.
    points, faces = [], []
    for a, b, c, d in quads:
        offset = len(points)
        points.extend(vertices[[a, b, c, d]])
        faces.extend([[offset, offset+1, offset+2], [offset, offset+2, offset+3]])
    return np.asarray(points), np.asarray(faces, dtype=np.uint32)


class ComparisonMesh:
    def __init__(self, prior, bodies):
        self.rest_nodes = np.concatenate([b.points for b in bodies])
        rests, skins, faces, groups = [], [], [], []
        vertex_offset = index_offset = 0
        for semantic, spacing, color in [(0, .001, [.66, .39, .43]), (1, .0006, [.95, .19, .43]), (2, .0012, [.12, .50, .78])]:
            rest, indices, binding = cluster_skin(prior[f'display_points_{semantic}'], prior[f'display_faces_{semantic}'],
                prior[f'display_nodes_{semantic}'], prior[f'display_weights_{semantic}'], len(self.rest_nodes), spacing)
            rests.append(rest); skins.append(binding); faces.append(indices+vertex_offset)
            groups.append(dict(semantic=semantic, first_vertex=vertex_offset, vertex_count=len(rest),
                first_index=index_offset, index_count=indices.size, color=color))
            vertex_offset += len(rest); index_offset += indices.size
        self.soft_count = vertex_offset
        self.binding = sparse.vstack(skins).tocsr()
        for lower, upper, color in [([-.02, -.004, -.003], [.02, .004, .003], [.98, .62, .08]),
                                     ([-.008, -.006, .003], [.008, .006, .053], [.82, .86, .91])]:
            rest, indices = box(lower, upper)
            rests.append(rest); faces.append(indices+vertex_offset)
            groups.append(dict(semantic=3, first_vertex=vertex_offset, vertex_count=len(rest),
                first_index=index_offset, index_count=indices.size, color=color))
            vertex_offset += len(rest); index_offset += indices.size
        self.rest = np.concatenate(rests)
        self.faces = np.concatenate(faces)
        self.metadata = dict(groups=groups, vertex_count=len(self.rest), index_count=self.faces.size,
            source='conventional indexed triangle MRI skin driven by actual Isaac FEM nodes',
            display_decimation='vertex clustering: brain 1 mm, tumour 0.6 mm, ventricles 1.2 mm',
            protocol='16 float header, then N x 6 float32 position and smooth normal; uint32 static indices')

    def pack(self, positions, pose, header):
        vertices = self.rest.copy()
        vertices[:self.soft_count] += self.binding@(np.concatenate(positions)-self.rest_nodes)
        yaw = 2*np.arctan2(pose[5], pose[6])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        vertices[self.soft_count:] = self.rest[self.soft_count:]@rotation.T+pose[:3]
        tri = vertices[self.faces]
        normals = np.zeros_like(vertices)
        face_normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        for i in range(3):
            np.add.at(normals, self.faces[:, i], face_normal)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        h = np.zeros(16, dtype=np.float32)
        h[:len(header)] = header
        h[0], h[3] = 2, len(vertices)
        return h.tobytes()+np.column_stack((vertices, normals)).astype(np.float32).tobytes()
