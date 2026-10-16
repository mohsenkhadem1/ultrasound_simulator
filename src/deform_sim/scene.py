"""USD appearance helpers; geometry here does not alter the contact model."""


def imaging_plane(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    mesh = UsdGeom.Mesh.Define(stage, '/World/AcousticPlane')
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    material = UsdShade.Material.Define(stage, '/World/AcousticPlaneMaterial')
    shader = UsdShade.Shader.Define(stage, '/World/AcousticPlaneMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.02, .65, .75))
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.13)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    line = UsdGeom.BasisCurves.Define(stage, '/World/AcousticPlaneOutline')
    line.CreateTypeAttr('linear')
    line.CreateCurveVertexCountsAttr([5])
    line.CreateWidthsAttr([.00035])
    line.SetWidthsInterpolation('constant')
    line.CreateDisplayColorAttr([Gf.Vec3f(.0, .8, .85)])
    return mesh, line


def add_probe_handle(stage, probe_path):
    """A tapered transducer casing, grip and cable above the rigid contact head."""
    import numpy as np
    from pxr import Gf, UsdGeom, Vt
    probe = UsdGeom.Cube.Get(stage, probe_path)
    root_path = probe_path+'/Casing'
    if stage.GetPrimAtPath(root_path):
        return
    # The collider is a scaled unit cube. Cancel that scale for metric children.
    scale = next(op.Get() for op in probe.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale)
    root = UsdGeom.Xform.Define(stage, root_path)
    root.AddScaleOp().Set(Gf.Vec3f(*[1/float(s) for s in scale]))
    vertices = []
    for z, width, thickness in [(0.003, .037, .009), (.014, .020, .015), (.049, .015, .012), (.055, .013, .010)]:
        vertices.extend([[-width/2, -thickness/2, z], [width/2, -thickness/2, z],
                         [width/2, thickness/2, z], [-width/2, thickness/2, z]])
    quads = [[0, 3, 2, 1], [12, 13, 14, 15]]
    for ring in range(3):
        for j in range(4):
            quads.append([ring*4+j, ring*4+(j+1)%4, (ring+1)*4+(j+1)%4, (ring+1)*4+j])
    casing = UsdGeom.Mesh.Define(stage, root_path+'/Shell')
    casing.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32)))
    casing.CreateFaceVertexCountsAttr([4]*len(quads))
    casing.CreateFaceVertexIndicesAttr(np.asarray(quads).ravel().tolist())
    casing.CreateDisplayColorAttr([Gf.Vec3f(.78, .80, .83)])
    casing.CreateSubdivisionSchemeAttr('catmullClark')
    for j in range(3):
        grip = UsdGeom.Cube.Define(stage, root_path+f'/Grip_{j}')
        grip.CreateSizeAttr(1.)
        grip.AddTranslateOp().Set(Gf.Vec3d(0, -.007, .025+j*.006))
        grip.AddScaleOp().Set(Gf.Vec3f(.016, .0015, .003))
        grip.CreateDisplayColorAttr([Gf.Vec3f(.12, .15, .18)])
    cable = UsdGeom.BasisCurves.Define(stage, root_path+'/Cable')
    cable.CreateTypeAttr('cubic')
    cable.CreateBasisAttr('bspline')
    cable.CreateCurveVertexCountsAttr([7])
    cable.CreatePointsAttr([[0, 0, .050], [0, 0, .056], [0, 0, .062], [.003, .002, .068], [.012, .004, .071], [.024, .007, .069], [.035, .010, .065]])
    cable.CreateWidthsAttr([.003])
    cable.SetWidthsInterpolation('constant')
    cable.CreateDisplayColorAttr([Gf.Vec3f(.10, .12, .15)])
