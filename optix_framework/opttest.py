import scipy.optimize
from scipy.optimize import least_squares
import numpy as np
from raw_utils import read_raw
from image_utils import rmse
from run_simulation import run, potato, vec_str
from image_utils import norm2_diff
from raw_utils import save_png
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os

os.chdir('./build/')
if not os.path.exists('../results'):
    os.mkdir('../results')

def burley_diffuse_approx(A):
    temp = albedo - np.array([0.33, 0.33, 0.33]);
    temp = temp*temp*temp*temp; 
    return np.array([3.5,3.5,3.5]) + 100.0 * temp;

def quotes(s):
    return "\"" + s +"\""
    
def run_sim(A,s):
    frames = 5000
    dipole = "APPROX_STANDARD_DIPOLE_BSSRDF"
    datapath = '../data'
    mesh = "/meshes/unit_sphere_2.obj"
    meshes = [mesh, "/meshes/circle_lights.obj"]
    materials = {mesh : potato} 
    materials[mesh].illum = 17
    dest_file = '../results/' + dipole.lower() + "_A_" + vec_str(A, '_') + "_s_" + vec_str(s, '_') + '.raw'
    bf = ["bssrdf/bssrdf_model", dipole, "bssrdf/approximate_A", quotes(vec_str(A)), "bssrdf/approximate_s", quotes(vec_str(s))] + ["light/background_constant_color", "\"0.0 0.0 0.0\""]
    run(frames, datapath, materials, meshes, bf , dest_file)
    return dest_file

def f(param, A, reference, plotdata):   # The rosenbrock function
    s = param
    dest_file = run_sim(A,s)
    dest = read_raw(dest_file)
    error = rmse(dest,reference)
    plotdata["plot"] += [[s[0], s[1], s[2], error]]
    plotdata["images"][str(error)] = dest_file
    print(error)
    return error

def f_lsq(param, A, reference, plotdata):   # The rosenbrock function
    s = param
    dest_file = run_sim(A,s)
    dest = read_raw(dest_file)
    error = dest - reference
    plotdata["plot"] += [[s[0], s[1], s[2], rmse(dest,reference)]]
    res = error.flatten()
    plotdata["images"][s.tostring()] = dest_file
    res[outlier] = 0.0
    print(res.max(), res.min())
    return res


albedo = potato.scattering / (potato.scattering + potato.absorption)
reduced_scattering = potato.scattering * (np.array([1,1,1]) - potato.g);
reduced_extinction  = reduced_scattering + potato.absorption;
transport = np.sqrt(3*potato.absorption* reduced_extinction)

initial_guess = burley_diffuse_approx(albedo) * transport

plotdata = { "plot" : [], "images" : {}}
reference_file = '../results/rendering_disc_lights_potato_volume_pt_50000_samples.raw'
reference = read_raw(reference_file)
outlier = np.argmax(reference.flatten())

plotdata2 = { "plot" : [], "images" : {}}

#xopt, fopt , iter , funcalls , warnflag, allvecs = scipy.optimize.fmin(f, initial_guess, args=(albedo, reference, plotdata2), full_output=1, retall=True, maxfun=1000)


opt_res = scipy.optimize.least_squares(f_lsq, initial_guess, diff_step =[0.001,0.001,0.001], bounds=([0,0,0], initial_guess*2), max_nfev =1000,  args=(albedo, reference, plotdata))
#%%
dest_folder = '../optimization_results/'
np.savetxt(dest_folder + 'result.txt', np.array(plotdata["plot"]))

import pickle
import shutil
opt = plotdata["images"][opt_res.x.tostring()]
dic_out = {}
dic_out["ref_name"] = os.path.basename(reference_file)
dic_out["opt_name"] = os.path.basename(opt)
shutil.copy(reference_file, dest_folder + dic_out["ref_name"])
shutil.copy(opt, dest_folder + dic_out["opt_name"])

def get_txt(a):
    return a[:-4] + '.txt'

shutil.copy(get_txt(reference_file), dest_folder + get_txt(dic_out["ref_name"]))
shutil.copy(get_txt(opt), dest_folder + get_txt(dic_out["opt_name"]))

with open(dest_folder + 'plotdata.pkl', "wb") as f:
    pickle.dump(dic_out, f, protocol=0)

#%%
data = np.array(plotdata["plot"])
np.savetxt('../optimization_results/results_gradient.txt', data)
#data = np.array(allvecs)
##
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dev = np.std(data[:,3])
avg = np.mean(data[:,3])
ax.scatter(data[:,1], data[:,2], data[:,3], c='r')
ax.set_zlim(avg - dev/2, avg + dev/2)
plt.savefig('data.pdf', bbox_inches='tight')

