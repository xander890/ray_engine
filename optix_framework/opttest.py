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
    frames = 3000
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
    plotdata["plot"] += [[s[0], s[1], s[2], error]]
       
    return error.flatten()


albedo = potato.scattering / (potato.scattering + potato.absorption)
reduced_scattering = potato.scattering * (np.array([1,1,1]) - potato.g);
reduced_extinction  = reduced_scattering + potato.absorption;
transport = np.sqrt(3*potato.absorption* reduced_extinction)

initial_guess = burley_diffuse_approx(albedo) * transport

plotdata = { "plot" : [], "images" : {}}
reference_file = '../results/rendering_disc_lights_potato_volume_pt_50000_samples.raw'
reference = read_raw(reference_file)

#xopt, fopt , iter , funcalls , warnflag, allvecs = scipy.optimize.fmin(f, initial_guess, args=(albedo, reference, plotdata), full_output=1, retall=True, maxfun=1000)
opt_res = scipy.optimize.least_squares(f_lsq, initial_guess, bounds=([0,0,0], initial_guess*2), max_nfev =1000,  args=(albedo, reference, plotdata))
#%%
np.savetxt('xopt.txt', xopt)
np.savetxt('fopt.txt', np.array([fopt]))
np.savetxt('allvecs.txt', allvecs)

#%%

opt = plotdata["images"][str(fopt)]
optimum = read_raw(opt)
save_png('optimum.png', optimum)
save_png('reference.png', reference)
diff = norm2_diff(optimum, reference)
#%%
plt.imshow(np.clip(diff,0,0.05), cmap='hot')
plt.colorbar()
#%%
data = np.array(plotdata["plot"])

#data = np.array(allvecs)
##
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dev = np.std(data[:,3])
avg = np.mean(data[:,3])
ax.scatter(data[:,1], data[:,2], data[:,3])
ax.set_zlim(avg - dev/2, avg + dev/2)
plt.savefig('data.pdf', bbox_inches='tight')

#plt.plot(data[:,0], label='x0')
#plt.plot(data[:,1], label='x1')
#plt.plot(data[:,2], label='x2')
#plt.ylabel('Value')
#plt.legend()
#plt.show()
#
