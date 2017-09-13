import sys
import subprocess
import os
materials = ["white_grapefruit", "potato", "ketchup", "mustard"]

ior = 1.3

illums = [12]
illum_names = {12 : "volume_pt", 14 : "screen_space_sampling", 17 : "point_cloud_sampling"}

scenes = {"rendering" : ["/meshes/unit_sphere_2.obj"], "rendering_disc_lights" : ["/meshes/unit_sphere_2.obj", "/meshes/circle_lights.obj"]}
scene_overrides = {"rendering" : [], "rendering_disc_lights" : ["light/background_constant_color", "\"0.1 0.1 0.1\""]}

frames = {"rendering" : 3000, "rendering_disc_lights" : 30000 }

os.chdir('./build/')
if not os.path.exists('../results'):
	os.mkdir('../results')
	
def create_mtl(mat_name, ior, illum):
	s = "newmtl " + mat_name + "\n"
	s += "Ni " + str(ior) + "\n"
	s += "illum " + str(illum) + "\n"
	return s

over_mat = "test.mtl"
for scene in scenes:
    meshes = scenes[scene]
    frame = frames[scene]
    print(meshes)
    for mat in materials:
        for illum in illums:
            with open(over_mat, 'w') as f:
                f.write(create_mtl(mat, ior, illum))
            res = '../results/'+scene+'_' + str(mat)  + "_" + str(illum_names[illum]) + "_" + str(frame) +  "_samples.raw" 
            print(res)
            if not os.path.exists(res):
                command = ['optix_framework.exe'] + meshes + ['-o', res, '-f', str(frame), '--material_override', over_mat]
                print(len(scene_overrides[scene]))
                if len(scene_overrides[scene]) > 0:
                    command += ["--parameter_override"] + scene_overrides[scene]
                print(" ".join(command))
                subprocess.call(command)