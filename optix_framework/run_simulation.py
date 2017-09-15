import sys
import subprocess
import os
import shutil
import numpy as np

class Material:
    def __init__(self, name, illum, ior, ka, kd, ks, absorption, scattering, asymmetry, scale):
        self.name = name
        self.illum = illum
        self.ior = ior
        self.absorption = np.array(absorption)
        self.scattering = np.array(scattering)
        self.g = np.array(asymmetry)
        self.scale = scale
        self.ka = np.array(ka)
        self.kd = np.array(kd)
        self.ks = np.array(ks)
        
        
    def __str__(self):
        s = "newmtl " + self.name + "\n"
        s += "Ka " + str(" ".join([str(a) for a in self.ka])) + "\n"
        s += "Kd " + str(" ".join([str(a) for a in self.kd])) + "\n"
        s += "Ks " + str(" ".join([str(a) for a in self.ks])) + "\n"
        s += "Ni " + str(self.ior) + "\n"
        s += "Sa " + str(" ".join([str(a) for a in self.absorption])) + "\n"
        s += "Ss " + str(" ".join([str(a) for a in self.scattering])) + "\n"
        s += "Sg " + str(" ".join([str(a) for a in self.g])) + "\n"
        s += "Sc " + str(self.scale) + "\n"
        s += "illum " + str(self.illum) + "\n"
        return s


def run(samples, datapath, materials, meshes, params, dest_file):
    for material in materials:
        material_to_write = datapath + material[:-4] + '.mtl'
        material_to_write_bk = material_to_write + '.bak'
        shutil.move(material_to_write, material_to_write_bk)
        with open(material_to_write, 'w') as f:
            f.write(str(materials[material]))
        if not os.path.exists(dest_file):
            command = ['optix_framework.exe'] + meshes + ['-o', dest_file, '-f', str(samples)]
            if len(params) > 0:
                command += ["--parameter_override"] + params
            print(" ".join(command))
            with open(dest_file[:-4] + '.log.txt', 'w') as f:
                subprocess.call(command, stdout=f, stderr=f)
        else:
            print( dest_file + " already exists, skipping.")

    for material in materials:
        material_to_write = datapath + material[:-4] + '.mtl'
        material_to_write_bk = material_to_write + '.bak'
        shutil.move(material_to_write_bk, material_to_write)
        
dipoles = ["STANDARD_DIPOLE_BSSRDF","DIRECTIONAL_DIPOLE_BSSRDF","APPROX_STANDARD_DIPOLE_BSSRDF","APPROX_DIRECTIONAL_DIPOLE_BSSRDF"]
potato = Material("potato", 12, ior=1.3, ka=[0,0,0], kd =[1,1,0], ks = [0,0,0], absorption=[0.0024,0.009,0.12], scattering=[0.68,0.70,0.55], asymmetry=[0,0,0], scale=7.0)

def vec_str(a, sep=" "):
    return sep.join([str(q) for q in a])


if __name__ == "__main__":
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


