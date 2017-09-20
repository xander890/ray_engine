import sys
import subprocess
import os
import shutil
import numpy as np
from materials import Material

def run(samples, datapath, materials, meshes, params, dest_file):
    if os.path.exists(dest_file):
        print( dest_file + " already exists, skipping.")
        return
    for material in materials:
        material_to_write = datapath + material[:-4] + '.mtl'
        material_to_write_bk = material_to_write + '.bak'
        shutil.move(material_to_write, material_to_write_bk)
        with open(material_to_write, 'w') as f:
            f.write(str(materials[material]))
        command = ['optix_framework.exe'] + meshes + ['-o', dest_file, '-f', str(samples)]
        if len(params) > 0:
            command += ["--parameter_override"] + params
        print(" ".join(command))
        with open(dest_file[:-4] + '.log.txt', 'w') as f:
            subprocess.call(command, stdout=f, stderr=f)


    for material in materials:
        material_to_write = datapath + material[:-4] + '.mtl'
        material_to_write_bk = material_to_write + '.bak'
        shutil.move(material_to_write_bk, material_to_write)
        
dipoles = ["STANDARD_DIPOLE_BSSRDF","DIRECTIONAL_DIPOLE_BSSRDF","APPROX_STANDARD_DIPOLE_BSSRDF","APPROX_DIRECTIONAL_DIPOLE_BSSRDF"]

def vec_str(a, sep=" "):
    return sep.join([str(q) for q in a])


if __name__ == "__main__":
    from materials import potato
    materials = {"/meshes/unit_sphere_2.obj" : potato} 
    illums = [12]
    illum_names = {12 : "volume_pt", 14 : "screen_space_sampling", 17 : "point_cloud_sampling"}
    scenes = {"rendering" : ["/meshes/unit_sphere_2.obj"], "rendering_disc_lights" : ["/meshes/unit_sphere_2.obj", "/meshes/circle_lights.obj"]}
    scene_overrides = {"rendering" : [], "rendering_disc_lights" : ["light/background_constant_color", "\"0.1 0.1 0.1\""]}
    frames = {"rendering" : 50, "rendering_disc_lights" : 50 }

    scenes_to_render = ["rendering"]
    A = [1.0, 1.0, 1.0]
    s = [1.0, 1.0, 1.0]
    os.chdir('./build/')
    if not os.path.exists('../results'):
        os.mkdir('../results')


    for dipole in dipoles:
        for scene in scenes_to_render:
            meshes = scenes[scene]
            frame = frames[scene]
            print(meshes)
            for illum in illums:
                for mat in materials.values():
                    mat.illum = illum
                res = '../results/'+ scene + "_" + str(illum_names[illum]) + "_" + str(frame) +  "_samples"+dipole.lower()+".raw" 
                bf = ["bssrdf/bssrdf_model", dipole, "bssrdf/approximate_A", vec_str(A), "bssrdf/approximate_s", vec_str(s)]
                run(frame, '../data', materials, meshes, scene_overrides[scene] + bf, res)


