import scipy.optimize
import numpy as np
from raw_utils import read_raw
from image_utils import rmse
from run_simulation import run, potato, vec_str
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

def f(param, A, reference):   # The rosenbrock function
    s = param
    dest_file = run_sim(A,s)
    dest = read_raw(dest_file)
    error = rmse(dest,reference)
    print(error)
    return error

albedo = potato.scattering / (potato.scattering + potato.absorption)
reduced_scattering = potato.scattering * (np.array([1,1,1]) - potato.g);
reduced_extinction  = reduced_scattering + potato.absorption;
transport = np.sqrt(3*potato.absorption* reduced_extinction)

initial_guess = burley_diffuse_approx(albedo) * transport


reference_file = '../results/rendering_disc_lights_potato_volume_pt_50000_samples.raw'
reference = read_raw(reference_file)

xopt, fopt , iter , funcalls , warnflag, allvecs = scipy.optimize.fmin(f, initial_guess, args=(albedo, reference), full_output=1, retall=True, maxfun=100)
#
#data = np.array(allvecs)
#
#import matplotlib.pyplot as plt
#plt.plot(data[:,0], label='x0')
#plt.plot(data[:,1], label='x1')
#plt.ylabel('Value')
#plt.legend()
#plt.show()

