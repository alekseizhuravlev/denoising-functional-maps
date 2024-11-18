import numpy as np
import trimesh
import torch
import os

from utils.geometry_util import laplacian_decomposition, get_operators




def preprocessing_pipeline(verts, faces, num_evecs, centering='bbox'):
    shape_dict = {
        'id': torch.tensor(-1),
        'verts': torch.tensor(verts).float(),
        'faces': torch.tensor(faces).long(),
    }
    
    # center the shape
    if centering == 'bbox':
        shape_dict['verts'] = center_bbox(shape_dict['verts'])
    elif centering == 'mean':
        shape_dict['verts'] = center_mean(shape_dict['verts'])
    else:
        raise RuntimeError(f'centering={centering} not recognized')
    
    # normalize vertices
    shape_dict['verts'] = normalize_face_area(shape_dict['verts'], shape_dict['faces'])
        
    # get spectral operators
    shape_dict = get_spectral_ops(
        shape_dict,
        num_evecs=num_evecs
        )
    
    return shape_dict
    


def center_mean(verts):
    """
    Center the vertices by subtracting the mean
    """
    verts -= torch.mean(verts, axis=0)
    return verts


def center_bbox(vertices):
    """
    Center the input mesh using its bounding box
    """
    bbox = torch.tensor([[torch.max(vertices[:,0]), torch.max(vertices[:,1]), torch.max(vertices[:,2])],
                     [torch.min(vertices[:,0]), torch.min(vertices[:,1]), torch.min(vertices[:,2])]])

    translation = (bbox[0] + bbox[1]) / 2
    translated_vertices = vertices - translation
    
    return translated_vertices


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


def inertia_transform(verts, faces):
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.apply_transform(mesh.principal_inertia_transform)
    
    return torch.tensor(mesh.vertices, dtype=torch.float32)
    


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)


    _, mass, L, evals, evecs, gradX, gradY = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    # evals = evals.unsqueeze(0)
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L
    item['gradX'] = gradX
    item['gradY'] = gradY

    return item

