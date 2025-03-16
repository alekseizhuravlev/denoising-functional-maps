import numpy as np
import trimesh
import torch

def plot_fmap(axis, C_12, l=None, h=None, title=None, show_grid=False, show_colorbar=False):
    
    if l is None or h is None:
        l = 0
        h = C_12.shape[-1]
    
    axis_img = axis.imshow(C_12[l:h, l:h], cmap='bwr', vmin=-1, vmax=1)
    
    if show_colorbar:
        figure = axis.figure
        figure.colorbar(axis_img, ax=axis)
    
    if title is not None:
        axis.set_title(f'{title}')

    axis.set_xticks(np.arange(-0.5, h - l, 1.0))
    axis.set_yticks(np.arange(-0.5, h - l, 1.0)) 
    
    if show_grid:
        axis.grid(which='both')    
    
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    
    
def plot_p2p_map(scene, verts_x, faces_x, verts_y, faces_y, p2p, axes_color_gradient=[0, 1],
                 base_cmap='jet'):
    
    # assert axes_color_gradient is a list or tuple
    assert isinstance(axes_color_gradient, (list, tuple)), "axes_color_gradient must be a list or tuple"
    assert verts_y.shape[0] == len(p2p), f"verts_y {verts_y.shape} and p2p {p2p.shape} must have the same length"
    
    # convert vertices and faces to tensors
    verts_x = torch.tensor(verts_x, dtype=torch.float32)
    verts_y = torch.tensor(verts_y, dtype=torch.float32)
    faces_x = torch.tensor(faces_x, dtype=torch.int64)
    faces_y = torch.tensor(faces_y, dtype=torch.int64)
    
    
    # normalize verts_x[:, 0] between 0 and 1
    # coords_x_norm = (verts_x[:, 0] - verts_x[:, 0].min()) / (verts_x[:, 0].max() - verts_x[:, 0].min())
    # coords_y_norm = (verts_x[:, 1] - verts_x[:, 1].min()) / (verts_x[:, 1].max() - verts_x[:, 1].min())
    # coords_z_norm = (verts_x[:, 2] - verts_x[:, 2].min()) / (verts_x[:, 2].max() - verts_x[:, 2].min())

    coords_x_norm = torch.zeros_like(verts_x)
    for i in range(3):
        coords_x_norm[:, i] = (verts_x[:, i] - verts_x[:, i].min()) / (verts_x[:, i].max() - verts_x[:, i].min())

    coords_interpolated = torch.zeros(verts_x.shape[0])
    for i in axes_color_gradient:
        coords_interpolated += coords_x_norm[:, i]
        
    # first colormap = interpolated y-axis values
    cmap = trimesh.visual.color.interpolate(coords_interpolated, base_cmap)
    
    # cmap = trimesh.visual.color.interpolate(verts_x[:, 0] + verts_x[:, 1], 'jet')
    
    # second colormap = first colormap values mapped to second mesh
    cmap2 = cmap[p2p].clip(0, 255)
    
    # diffuse material
    material=trimesh.visual.material.SimpleMaterial(
        image=None,
        diffuse=[245] * 4,
    )

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=verts_x, faces=faces_x, validate=True)
    mesh1.visual.material = material
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)
    
    
    
    # add the second mesh
    mesh2 = trimesh.Trimesh(vertices=verts_y + np.array([1, 0, 0]), faces=faces_y, validate=True)
    mesh2.visual.material = material
    mesh2.visual.vertex_colors = cmap2[:len(mesh2.vertices)]
    scene.add_geometry(mesh2)
    
    scene.add_geometry(trimesh.creation.axis(origin_size=0.05))

    return scene