import inspect
import os
import sys
import argparse
import subprocess

#files = ['./models/01_solo_combined_triangulated.obj','./models/glass_sphere_spheres.obj','./models/glass_sphere_mesh_cleaned.obj']
#files = ['./models/01_solo_combined_triangulated.obj','./models/JDS_glass_teapot_bottom_mesh_cleaned_manual_transformed.obj','./models/JDS_glass_teapot_bottom_spheres_cleaned_transformed.obj','./models/JDS_glass_teapot_top_mesh_cleaned_manual_transformed.obj','./models/JDS_glass_teapot_top_spheres_cleaned_transformed.obj']
files = ['./models/01_solo_combined_triangulated.obj','./models/glass_bowl_bottom_mesh_cleaned_manual_transformed.obj','./models/glass_bowl_top_mesh_cleaned_manual_transformed.obj','./models/glass_bowl_top_spheres_cleaned_transformed.obj','./models/glass_bowl_bottom_spheres_cleaned_transformed.obj']
#files = ['./models/01_solo_combined_triangulated.obj','./models/glass_teapot_decimated_smoothed_normal.obj','./models/JDS_glass_teapot_bottom_spheres_cleaned_transformed.obj','./models/JDS_glass_teapot_top_mesh_cleaned_manual_transformed.obj','./models/JDS_glass_teapot_top_spheres_cleaned_transformed.obj']
#files = ['./models/01_solo_combined_triangulated.obj','./models/JDS_glass_teapot_top_mesh_cleaned_manual_transformed_no_markers_taubinsmoothed.obj','./models/JDS_glass_teapot_top_mesh_cleaned_manual_transformed.obj','./models/JDS_glass_teapot_top_spheres_cleaned_transformed.obj']
#files = ['./models/01_solo_combined_triangulated.obj','./models/glass_sphere_mesh_cleaned.obj']

def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False): # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)

os.chdir('optix_framework')	
command = 'optix_framework.exe'

subprocess.call([command] + files)
