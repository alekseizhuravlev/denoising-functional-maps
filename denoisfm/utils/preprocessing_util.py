import os

import numpy as np
import torch
import trimesh
from .geometry_util import get_operators, laplacian_decomposition
from .shape_util import compute_geodesic_distmat


def preprocessing_pipeline(verts, faces, num_evecs, compute_distmat=False, lb_cache_dir=None):
    shape_dict = {
        "id": torch.tensor(-1),
        "verts": verts if isinstance(verts, torch.Tensor) else torch.tensor(verts, dtype=torch.float),
        "faces": faces if isinstance(faces, torch.Tensor) else torch.tensor(faces, dtype=torch.long),
    }

    shape_dict["verts"] = center_mean(shape_dict["verts"])

    # normalize vertices
    shape_dict["verts"] = normalize_face_area(shape_dict["verts"], shape_dict["faces"])

    # get spectral operators
    shape_dict = get_spectral_ops(shape_dict, num_evecs=num_evecs, cache_dir=lb_cache_dir)
    
    if compute_distmat:
        shape_dict['dist'] = torch.tensor(
            compute_geodesic_distmat(shape_dict['verts'].numpy(), shape_dict['faces'].numpy()),
            dtype=torch.float32    
        )
    
    return shape_dict


def center_mean(verts):
    """
    Center the vertices by subtracting the mean
    """
    verts -= torch.mean(verts, axis=0)
    return verts


def normalize_face_area(verts, faces):
    """
    Calculate the square root of the area through laplacian decomposition
    Normalize the vertices by it
    """
    verts = np.array(verts)
    faces = np.array(faces)

    old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
    verts /= old_sqrt_area

    return torch.tensor(verts)


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    _, mass, L, evals, evecs, gradX, gradY = get_operators(
        item["verts"], item.get("faces"), k=num_evecs, cache_dir=cache_dir
    )
    evecs_trans = evecs.T * mass[None]
    item["evecs"] = evecs[:, :num_evecs]
    item["evecs_trans"] = evecs_trans[:num_evecs]
    item["evals"] = evals[:num_evecs]
    item["mass"] = mass
    item["L"] = L
    item["gradX"] = gradX
    item["gradY"] = gradY

    return item
