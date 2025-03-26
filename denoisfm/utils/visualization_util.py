import numpy as np
import torch
import trimesh


def plot_fmap(
    axis, C_12, l=None, h=None, title=None, show_grid=False, show_colorbar=False
):
    """
    Plots a matrix as an image on the given axis using a colormap.

    Args:
        axis (matplotlib.axes.Axes): The axis object where the image will be plotted.
        C_12 (numpy.ndarray): A 2D matrix to be visualized, typically a functional map or a conditioning matrix.
        l (int, optional): The starting index for the portion of the matrix to display. Defaults to 0.
        h (int, optional): The ending index for the portion of the matrix to display. Defaults to the size of the matrix.
        title (str, optional): The title of the plot. If None, no title is set.
        show_grid (bool, optional): Whether to show the gridlines on the plot. Defaults to False.
        show_colorbar (bool, optional): Whether to display a color bar. Defaults to False.

    Returns:
        None: Displays the image on the provided axis.

    Notes:
        The colormap used is 'bwr' (blue-white-red) with values clipped between -1 and 1.
    """

    if l is None or h is None:
        l = 0
        h = C_12.shape[-1]

    axis_img = axis.imshow(C_12[l:h, l:h], cmap="bwr", vmin=-1, vmax=1)

    if show_colorbar:
        figure = axis.figure
        figure.colorbar(axis_img, ax=axis)

    if title is not None:
        axis.set_title(f"{title}")

    axis.set_xticks(np.arange(-0.5, h - l, 1.0))
    axis.set_yticks(np.arange(-0.5, h - l, 1.0))

    if show_grid:
        axis.grid(which="both")

    axis.set_xticklabels([])
    axis.set_yticklabels([])


def plot_p2p_map(
    scene,
    verts_1,
    faces_1,
    verts_2,
    faces_2,
    Pi_21,
    axes_color_gradient=[0, 1],
    base_cmap="jet",
):
    """
    Plots a point-to-point correspondence map between two 3D meshes in a scene, where colors are interpolated
    based on vertex positions and the point correspondence is defined by Pi_21.

    Args:
        scene (trimesh.Scene): The trimesh scene object where the meshes will be added.
        verts_1 (numpy.ndarray or torch.Tensor): The vertices of the first mesh, shape (V1, 3).
        faces_1 (numpy.ndarray or torch.Tensor): The faces of the first mesh, shape (F1, 3).
        verts_2 (numpy.ndarray or torch.Tensor): The vertices of the second mesh, shape (V2, 3).
        faces_2 (numpy.ndarray or torch.Tensor): The faces of the second mesh, shape (F2, 3).
        Pi_21 (numpy.ndarray or torch.Tensor): The point-to-point correspondence, shape (V2,).
        axes_color_gradient (list or tuple, optional): The axes along which to interpolate the color gradient,
                                                       e.g. 0 for x-axis and 1 for y-axis. Defaults to [0, 1].
        base_cmap (str, optional): The colormap to use for coloring. Defaults to 'jet'.

    Returns:
        trimesh.Scene: The updated scene object containing the two meshes with corresponding colors based on the
                        point-to-point correspondence.
    """

    # assert axes_color_gradient is a list or tuple
    assert isinstance(axes_color_gradient, (list, tuple)), (
        "axes_color_gradient must be a list or tuple"
    )
    assert verts_2.shape[0] == len(Pi_21), (
        f"verts_2 {verts_2.shape} and Pi_21 {Pi_21.shape} must have the same length"
    )

    # convert vertices and faces to tensors
    verts_1 = torch.tensor(verts_1, dtype=torch.float32)
    verts_2 = torch.tensor(verts_2, dtype=torch.float32)
    faces_1 = torch.tensor(faces_1, dtype=torch.int64)
    faces_2 = torch.tensor(faces_2, dtype=torch.int64)

    # normalize the coordinates of the first mesh
    coords_1_norm = torch.zeros_like(verts_1)
    for i in range(3):
        coords_1_norm[:, i] = (verts_1[:, i] - verts_1[:, i].min()) / (
            verts_1[:, i].max() - verts_1[:, i].min()
        )

    # select the axes which will be used to interpolate the color gradient
    # e.g. only x and y axes
    coords_interpolated = torch.zeros(verts_1.shape[0])
    for i in axes_color_gradient:
        coords_interpolated += coords_1_norm[:, i]

    # first colormap = interpolated y-axis values
    cmap = trimesh.visual.color.interpolate(coords_interpolated, base_cmap)

    # second colormap = first colormap values mapped to second mesh
    cmap2 = cmap[Pi_21].clip(0, 255)

    # diffuse material
    material = trimesh.visual.material.SimpleMaterial(
        image=None,
        diffuse=[245] * 4,
    )

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=verts_1, faces=faces_1, validate=True)
    mesh1.visual.material = material
    mesh1.visual.vertex_colors = cmap[: len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)

    # add the second mesh
    mesh2 = trimesh.Trimesh(
        vertices=verts_2 + np.array([1, 0, 0]), faces=faces_2, validate=True
    )
    mesh2.visual.material = material
    mesh2.visual.vertex_colors = cmap2[: len(mesh2.vertices)]
    scene.add_geometry(mesh2)

    scene.add_geometry(trimesh.creation.axis(origin_size=0.05))

    return scene
