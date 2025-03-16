import numpy as np
import pymeshlab
import torch
import trimesh

from denoisfm.utils import fmap_util

ms = pymeshlab.MeshSet()


def remesh_pipeline(verts_orig, faces_orig, config):
    
    # randomly choose the remeshing type
    remesh_type = np.random.choice(['isotropic', 'anisotropic'], p=[1-config["anisotropic"]["probability"], config["anisotropic"]["probability"]])
    
    if remesh_type == 'isotropic':
        
        # isotropic remeshing
        simplify_strength = np.random.uniform(config["isotropic"]["simplify_strength_min"], config["isotropic"]["simplify_strength_max"])
        verts, faces = remesh_simplify_iso(
            verts_orig,
            faces_orig,
            remesh=config["isotropic"]["remesh"],
            simplify_strength=simplify_strength,
        )
    else:
        
        # anisotropic remeshing
        fraction_to_simplify = np.random.uniform(config["anisotropic"]["fraction_to_simplify_min"], config["anisotropic"]["fraction_to_simplify_max"])
        simplify_strength = np.random.uniform(config["anisotropic"]["simplify_strength_min"], config["anisotropic"]["simplify_strength_max"])
        
        verts, faces = remesh_simplify_anis(
            verts_orig,
            faces_orig,
            remesh=config["isotropic"]["remesh"],
            fraction_to_simplify=fraction_to_simplify,
            simplify_strength=simplify_strength,
        )
        
    # correspondence by a nearest neighbor search
    corr = fmap_util.nn_query(
        verts,
        verts_orig,
        )
    
    return verts, faces, corr


def remesh_simplify_iso(
    verts,
    faces,
    remesh=True,
    simplify_strength=0.3,
):
    mesh = pymeshlab.Mesh(verts, faces)
    ms.add_mesh(mesh)

    if remesh:
        ms.meshing_isotropic_explicit_remeshing()
        
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=simplify_strength,
    )
        
    v_qec = torch.tensor(
        ms.current_mesh().vertex_matrix(), dtype=torch.float32
    )
    f_qec = torch.tensor(
        ms.current_mesh().face_matrix(), dtype=torch.long
    )
    
    ms.clear()
    
    return v_qec, f_qec


def remesh_simplify_anis(
    verts,
    faces,
    remesh=True,
    fraction_to_simplify=0.3,
    simplify_strength=0.3,
    weighted_by='face_count',
    ):
    
    assert weighted_by in ['area', 'face_count']   
    
    mesh = pymeshlab.Mesh(verts, faces)
    ms.add_mesh(mesh)
    
    # isotropic remeshing
    if remesh:
        ms.meshing_isotropic_explicit_remeshing()
        
    # mesh after remeshing   
    v_r = ms.current_mesh().vertex_matrix()
    f_r = ms.current_mesh().face_matrix()
    
    if weighted_by == 'area':
        # face area
        mesh_r = trimesh.Trimesh(v_r, f_r)
        area_faces = mesh_r.area_faces
        total_area_faces = area_faces.sum()

        # choose a random face, with probability proportional to its area
        rand_idx = np.random.choice(len(area_faces), p=area_faces / total_area_faces)
        
    elif weighted_by == 'face_count':
        # choose a random face
        rand_idx = np.random.randint(0, len(f_r))

    # select the face
    ms.set_selection_none()
    ms.compute_selection_by_condition_per_face(
        condselect= f'(fi == {rand_idx})'
    )
    
    # select the simplification area by dilatation
    for dil_iter in range(100):
        
        # stopping criterion
        if weighted_by == 'area':
            selected_area = sum(area_faces[ms.current_mesh().face_selection_array()])
            if selected_area >= total_area_faces * fraction_to_simplify:
                break
            
        elif weighted_by == 'face_count':
            selected_faces = sum(ms.current_mesh().face_selection_array())
            if selected_faces >= len(f_r) * fraction_to_simplify:
                break
        ms.apply_selection_dilatation()
        

    selected_faces = ms.current_mesh().face_selection_array()

    # simplify the mesh
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(sum(ms.current_mesh().face_selection_array()) * simplify_strength),
        selected=True
    )

    # get the vertices and faces
    v_qec = torch.tensor(
        ms.current_mesh().vertex_matrix(), dtype=torch.float32
    )
    f_qec = torch.tensor(
        ms.current_mesh().face_matrix(), dtype=torch.long
    )
    
    ms.clear()
    
    return v_qec, f_qec
    
