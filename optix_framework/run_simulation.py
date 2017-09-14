import sys
import subprocess
import os
import shutil

class Material:
    def __init__(self, name, illum, ior, absorption, scattering, asymmetry, scale):
        self.name = name
        self.illum = illum
        self.ior = ior
        self.absorption = absorption
        self.scattering = scattering
        self.g = asymmetry
        self.scale = scale
    def __str__(self):
        s = "newmtl " + self.name + "\n"
        s += "Ni " + str(self.ior) + "\n"
        s += "Sa " + str(" ".join([str(a) for a in self.absorption])) + "\n"
        s += "Ss " + str(" ".join([str(a) for a in self.scattering])) + "\n"
        s += "Sg " + str(" ".join([str(a) for a in self.g])) + "\n"
        s += "Sc " + str(self.scale) + "\n"
        s += "illum " + str(self.illum) + "\n"
        return s
        
        
pot = Material("potato", 12, ior=1.3, absorption=[0.0024,0.009,0.12], scattering=[0.68,0.70,0.55], asymmetry=[0,0,0], scale=100.0)
materials = [pot]

illums = [14]
illum_names = {12 : "volume_pt", 14 : "screen_space_sampling", 17 : "point_cloud_sampling"}

scenes = {"rendering" : ["/meshes/unit_sphere_2.obj"], "rendering_disc_lights" : ["/meshes/unit_sphere_2.obj", "/meshes/circle_lights.obj"]}
scene_overrides = {"rendering" : [], "rendering_disc_lights" : ["light/background_constant_color", "\"0.1 0.1 0.1\""]}

material_to_write = "../data/meshes/unit_sphere.mtl"
material_to_write_bk = material_to_write + '.bak'

frames = {"rendering" : 50, "rendering_disc_lights" : 50 }

os.chdir('./build/')
if not os.path.exists('../results'):
    os.mkdir('../results')

shutil.move(material_to_write, material_to_write_bk)

for scene in scenes:
    meshes = scenes[scene]
    frame = frames[scene]
    print(meshes)
    for mat in materials:
        for illum in illums:
            mat.illum = illum
            with open(material_to_write, 'w') as f:
                f.write(str(mat))
            res = '../results/'+scene+'_' + mat.name + "_" + str(illum_names[illum]) + "_" + str(frame) +  "_samples.raw" 
            print(res)
            if not os.path.exists(res):
                command = ['optix_framework.exe'] + meshes + ['-o', res, '-f', str(frame)]
                print(len(scene_overrides[scene]))
                if len(scene_overrides[scene]) > 0:
                    command += ["--parameter_override"] + scene_overrides[scene]
                print(" ".join(command))
                subprocess.call(command)
            else:
                print(res + " already exists, skipping.")

shutil.move(material_to_write_bk, material_to_write)