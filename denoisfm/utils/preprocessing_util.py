import os

import numpy as np
import torch

from .geometry_util import get_operators, laplacian_decomposition
from .shape_util import compute_geodesic_distmat


def preprocessing_pipeline(
    verts, faces, num_evecs, compute_distmat=False, lb_cache_dir=None
):
    """
    Preprocesses a 3D mesh by centering the vertices, normalizing the face area,
    and computing spectral operators.

    Args:
        verts (torch.Tensor or np.ndarray): Vertices of the mesh, shape (V, 3), where V is the number of vertices.
        faces (torch.Tensor or np.ndarray): Faces of the mesh, shape (F, 3), where F is the number of faces.
        num_evecs (int): Number of eigenvectors to compute for the spectral decomposition.
        compute_distmat (bool, optional): Whether to compute the geodesic distance matrix. Defaults to False.
        lb_cache_dir (str, optional): Directory to cache the spectral operators if provided. Defaults to None.

    Returns:
        dict: A dictionary containing preprocessed mesh data:
            - "verts": Centered and normalized vertices (torch.Tensor).
            - "faces": Input faces (torch.Tensor).
            - "evecs": Computed eigenvectors (torch.Tensor).
            - "evecs_trans": Transposed eigenvectors multiplied by the vertex-area matrix (torch.Tensor).
            - "evals": Computed eigenvalues (torch.Tensor).
            - "mass": Vertex-area matrix (torch.Tensor).
            - "L": Laplacian matrix (torch.Tensor).
            - "gradX": Gradient X (torch.Tensor).
            - "gradY": Gradient Y (torch.Tensor).
            - "dist" (optional): Geodesic distance matrix (torch.Tensor) if compute_distmat is True.
    """

    shape_dict = {
        "id": torch.tensor(-1),
        "verts": verts
        if isinstance(verts, torch.Tensor)
        else torch.tensor(verts, dtype=torch.float),
        "faces": faces
        if isinstance(faces, torch.Tensor)
        else torch.tensor(faces, dtype=torch.long),
    }

    shape_dict["verts"] = center_mean(shape_dict["verts"])

    # normalize vertices
    shape_dict["verts"] = normalize_face_area(shape_dict["verts"], shape_dict["faces"])

    # get spectral operators
    shape_dict = get_spectral_ops(
        shape_dict, num_evecs=num_evecs, cache_dir=lb_cache_dir
    )

    if compute_distmat:
        shape_dict["dist"] = torch.tensor(
            compute_geodesic_distmat(
                shape_dict["verts"].numpy(), shape_dict["faces"].numpy()
            ),
            dtype=torch.float32,
        )

    return shape_dict


def center_mean(verts):
    """
    Centers the vertices by subtracting the mean along each axis.
    """
    verts -= torch.mean(verts, axis=0)
    return verts


def normalize_face_area(verts, faces):
    """
    Normalizes the vertices, making the total face area equal to 1.

    Args:
        verts (torch.Tensor or np.ndarray): Vertices of the mesh, shape (V, 3).
        faces (torch.Tensor or np.ndarray): Faces of the mesh, shape (F, 3).

    Returns:
        torch.Tensor: Normalized vertices, shape (V, 3).
    """
    verts = np.array(verts)
    faces = np.array(faces)

    old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
    verts /= old_sqrt_area

    return torch.tensor(verts)


def get_spectral_ops(shape_dict, num_evecs, cache_dir=None):
    """
    Computes the spectral operators (eigenvectors, eigenvalues, Laplacian, mass matrix, gradients)
    for the given vertices and faces of the mesh.

    Args:
        shape_dict (dict): A dictionary containing the mesh data with keys "verts" and "faces".
        num_evecs (int): Number of eigenvectors to compute for the spectral decomposition.
        cache_dir (str, optional): Directory to cache the spectral operators if provided. Defaults to None.

    Returns:
        dict: The input dictionary with additional keys:
            - "evecs": Eigenvectors (torch.Tensor).
            - "evecs_trans": Transposed eigenvectors (torch.Tensor).
            - "evals": Eigenvalues (torch.Tensor).
            - "mass": Vertex-area matrix (torch.Tensor).
            - "L": Laplacian matrix (torch.Tensor).
            - "gradX": Gradient X (torch.Tensor).
            - "gradY": Gradient Y (torch.Tensor).
    """

    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    _, mass, L, evals, evecs, gradX, gradY = get_operators(
        shape_dict["verts"], shape_dict.get("faces"), k=num_evecs, cache_dir=cache_dir
    )
    evecs_trans = evecs.T * mass[None]
    shape_dict["evecs"] = evecs[:, :num_evecs]
    shape_dict["evecs_trans"] = evecs_trans[:num_evecs]
    shape_dict["evals"] = evals[:num_evecs]
    shape_dict["mass"] = mass
    shape_dict["L"] = L
    shape_dict["gradX"] = gradX
    shape_dict["gradY"] = gradY

    return shape_dict
