import sys
import subprocess
import os
materials = ["white_grapefruit","potato","ketchup", "coffee", "mustard"]

ior = 1.3

illums = [12]
illum_names = {12 : "volume_pt", 14 : "screen_space_sampling", 17 : "point_cloud_sampling"}

mesh = "/meshes/unit_sphere_2.obj"

frames = 1000

os.chdir('./build/')
if not os.path.exists('../results'):
	os.mkdir('../results')
	
def create_mtl(mat_name, ior, illum):
	s = "newmtl " + mat_name + "\n"
	s += "Ni " + str(ior) + "\n"
	s += "illum " + str(illum) + "\n"
	return s

over_mat = "test.mtl"	
for mat in materials:
	for illum in illums:
		with open(over_mat, 'w') as f:
			f.write(create_mtl(mat, ior, illum))
        res = '../results/rendering_' + str(mat)  + "_" + str(illum_names[illum]) + "_" + str(frames) +  "_samples.raw" 
        if not os.path.exists(res):
            command = ['optix_framework.exe', mesh, '-o', res, '-f', str(frames), '--material_override', over_mat]
            subprocess.call(command)