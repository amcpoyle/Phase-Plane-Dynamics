import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from scipy.integrate import odeint

# read in parameters from an excel file

class Vehicle():
    vehicle_params = {'g': None, 'vehicleMass': None, 'trackwidth': None, 'wheelbase': None,
                      'b': None, 'a': None, 'h': None, 'rho_air': None,
                      'CdA': None, 'CLfA': None, 'CLrA': None, 'gamma': None,
                      'roll_stiffness': None, 'P_max': None, 'r_R': None,
                      'c_alpha_f': None, 'c_alpha_r': None, 'mu': None}
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_excel(self.data_path)
        for param in Vehicle.vehicle_params.keys():
            value = df[df['Parameter'] == param]['Value']
            Vehicle.vehicle_params[param] = value.iloc[0]

class TireModel():
    tire_params = {'ref_load': None, 'pCx1': None, 'pDx1': None, 'pDx2': None, 'pEx1': None,
                   'pKx1': None, 'pKx3': None, 'lambda_mux': None,
                   'pCy1': None, 'pDy1': None, 'pDy2': None, 'pEy1': None, 'pKy1': None,
                   'pKy2': None, 'lambda_muy': None,
                   'd1_fy': None, 'd2_fy': None, 'b_fy': None, 'c_fy': None}

    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_excel(self.data_path)
        for param in TireModel.tire_params.keys():
            value = df[df['Parameter'] == param]['Value']
            TireModel.tire_params[param] = value.iloc[0]


def mf_fy(alpha, Fz, tire_coefs):
    d = (tire_coefs['d1_fy'] + tire_coefs['d2_fy']/1000 * Fz)*Fz
    fy = d*np.sin(tire_coefs['c_fy']*np.atan(tire_coefs['b_fy']*alpha))
    return fy

def mf_fx_fy(kappa, alpha, Fz, tire_coefs):

    error_eps = 1e-6
    # calculate the coefs
    dfz = (Fz - tire_coefs['ref_load'])/tire_coefs['ref_load']
    Kx = Fz*tire_coefs['pKx1']*ca.exp(tire_coefs['pKx3']*dfz)
    Ex = tire_coefs['pEx1']
    Dx = (tire_coefs['pDx1'] + tire_coefs['pDx2']*dfz)*tire_coefs['lambda_mux']
    Cx = tire_coefs['pCx1']
    Bx = Kx/(Cx*Dx*Fz)
    
    Ky = tire_coefs['ref_load']*tire_coefs['pKy1']*ca.sin(2*ca.atan(Fz/(tire_coefs['pKy2']*tire_coefs['ref_load'])))
    Ey = tire_coefs['pEy1']
    Dy = (tire_coefs['pDy1'] + tire_coefs['pDy2']*dfz)*tire_coefs['lambda_muy']
    Cy = tire_coefs['pCy1']
    By = Ky/(Cy*Dy*Fz)

    # magic formula
    sig_x = kappa/(1 + kappa)
    sig_y = alpha/(1 + kappa)
    sig = ca.sqrt((sig_x**2) + (sig_y**2))

    Fx = Fz*(sig_x/(sig + error_eps))*Dx*ca.sin(Cx * ca.atan(Bx*sig - Ex*(Bx*sig - ca.atan(Bx*sig))))
    Fy = Fz*(sig_y/(sig + error_eps))*Dy*ca.sin(Cy*ca.atan(By*sig - Ey*(By*sig - ca.atan(By*sig))))


    return Fx, Fy

def normal_loads(ax, ay, u, vehicle_coefs):
    FLf = 0.5*vehicle_coefs['CLfA']*vehicle_coefs['rho_air']*(u**2)
    FLr = 0.5*vehicle_coefs['CLrA']*vehicle_coefs['rho_air']*(u**2)
    vehicleMass = vehicle_coefs['vehicleMass']
    b = vehicle_coefs['b']
    a = vehicle_coefs['a']
    h = vehicle_coefs['h']
    trackwidth = vehicle_coefs['trackwidth']
    roll_stiffness = vehicle_coefs['roll_stiffness']
    g = vehicle_coefs['g']

    Nfl = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nfr = 0.5*vehicleMass*g*(b/(a+b)) - 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*roll_stiffness + 0.5*FLf        
    Nrl = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) + vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr        
    Nrr = 0.5*vehicleMass*g*(a/(a+b)) + 0.5*vehicleMass*ax*(h/(a+b)) - vehicleMass*ay*(h/trackwidth)*(1 - roll_stiffness) + 0.5*FLr

    Nfl = ca.fmax(Nfl, 1e-3)
    Nfr = ca.fmax(Nfr, 1e-3)
    Nrl = ca.fmax(Nrl, 1e-3)
    Nrr = ca.fmax(Nrr, 1e-3)


    return Nfl, Nfr, Nrl, Nrr

def normal_forces(vehicle_coefs):
    Fz_front = (vehicle_coefs['vehicleMass']*vehicle_coefs['g']*vehicle_coefs['b'])/(vehicle_coefs['a'] + vehicle_coefs['b'])

    Fz_rear = (vehicle_coefs['vehicleMass']*vehicle_coefs['g']*vehicle_coefs['a'])/(vehicle_coefs['a'] + vehicle_coefs['b'])
    return Fz_front, Fz_rear

def lateral_brush(Fz, C_alpha, alpha, vehicle_coefs, mu):
    Fz_front, Fz_rear = Fz
    C_alpha_f, C_alpha_r = C_alpha
    alpha_f, alpha_r = alpha
    # mu = estimated friction of driving surface
    alpha_sl_front = np.atan((3*mu*Fz_front)/(C_alpha_f))
    alpha_sl_rear = np.atan((3*mu*Fz_rear)/(C_alpha_r))

    if abs(alpha_f) < alpha_sl_front:
        fy_front = -C_alpha_f*np.tan(alpha_f) + ((C_alpha_f**2)/(3*mu*Fz_front))*abs(np.tan(alpha_f))*np.tan(alpha_f) - ((C_alpha_f**3)/(27*(mu**2)*(Fz_front**2)))*(np.tan(alpha_f))**3
    else:
        fy_front = -mu*Fz_front*np.sign(alpha_f)


    if abs(alpha_r) < alpha_sl_rear:
        fy_rear = -C_alpha_r*np.tan(alpha_r) + ((C_alpha_r**2)/(3*mu*Fz_rear))*abs(np.tan(alpha_r))*np.tan(alpha_r) - ((C_alpha_r**3)/(27*(mu**2)*(Fz_rear**2)))*(np.tan(alpha_r))**3
    else:
        fy_rear = -mu*Fz_rear*np.sign(alpha_r)

    return fy_front, fy_rear

def linear_model(C_alpha, alpha):
    C_alpha_f, C_alpha_r = C_alpha
    alpha_f, alpha_r = alpha
    fy_front = -C_alpha_f*alpha_f
    fy_rear = -C_alpha_r*alpha_r
    return fy_front, fy_rear
###################################################

# equations for fsolve
def eqns(X):
    beta, r = X

    Fz_front, Fz_rear = normal_forces(vehicle_coefs)
    
    alpha_front = beta + (vehicle_coefs['a']*r)/vx - delta
    alpha_rear = beta - (vehicle_coefs['b']*r)/vx
    
    Fz = (Fz_front, Fz_rear)
    C_alpha = (vehicle_coefs['c_alpha_f'], vehicle_coefs['c_alpha_r'])
    alpha = (alpha_front, alpha_rear)
    
    fy_front = mf_fy(alpha_front, Fz_front, tire_coefs)
    fy_rear = mf_fy(alpha_rear, Fz_rear, tire_coefs)
    # fy_front, fy_rear = lateral_brush(Fz, C_alpha, alpha, vehicle_coefs, mu)
    # fy_front, fy_rear = linear_model(C_alpha, alpha)
    
    Izz = (1/12)*vehicle_coefs['vehicleMass']*(vehicle_coefs['trackwidth']**2 + vehicle_coefs['wheelbase']**2)
    
    beta_dot = (fy_front + fy_rear)/(vehicle_coefs['vehicleMass']*vx) - r
    r_dot = (vehicle_coefs['a']*fy_front - vehicle_coefs['b']*fy_rear)/Izz
    # magnitude = np.sqrt(beta_dot**2 + r_dot**2) + 1e-8
    # beta_dot = beta_dot/magnitude
    # r_dot = r_dot/magnitude

    return [beta_dot, r_dot]

def jacobian(eqns, x, eps=1e-6):
    x = np.asarray(x, dtype=float) # state becomes a matrix
    n = len(x) # dimension of x
    J = np.zeros((n, n)) # jacobian matrix
    f0 = np.asarray(eqns(x)) # evaluating eqns at fixed point

    for i in range(n):
        # perturb 1 variable
        dx = np.zeros(n)
        dx[i] = eps
        f1 = np.asarray(eqns(x+dx))
        J[:, i] = (f1-f0)/eps

    return J

def classify_fixed_pt(fp):
    J = jacobian(eqns, fp)
    eigvals = np.linalg.eigvals(J)

    real = np.real(eigvals)
    
    # classifying stability
    if np.all(real < 0):
        stability = 'stable'
    elif np.all(real > 0):
        stability = 'unstable'
    elif np.any(real > 0) and np.any(real < 0):
        stability = 'saddle'
    else:
        stability = 'marginal'

    return eigvals, stability

def find_fixed_pts(beta_max, r_max):
    fixed_pts = []

    beta_range = np.linspace(-beta_max, beta_max, 20)
    r_range = np.linspace(-r_max, r_max, 20)
    
    for beta_guess in beta_range:
        for r_guess in r_range:
            x0 = [beta_guess, r_guess]
            sol = fsolve(eqns, x0, full_output=True)
            sol = np.round(sol[0], decimals=2)

            if abs(sol[0]) > beta_max or abs(sol[1]) > r_max:
                continue

            if any(np.all(sol == fp) for fp in fixed_pts):
                continue
            else:
                fixed_pts.append(sol)

    return fixed_pts

def r_dot_eqn(X, beta):
    r = X

    Fz_front, Fz_rear = normal_forces(vehicle_coefs)
    
    alpha_front = beta + (vehicle_coefs['a']*r)/vx - delta
    alpha_rear = beta - (vehicle_coefs['b']*r)/vx
    
    Fz = (Fz_front, Fz_rear)
    C_alpha = (C_alpha_front, C_alpha_rear)
    alpha = (alpha_front, alpha_rear)
    
    fy_front = mf_fy(alpha_front, Fz_front, tire_coefs)
    fy_rear = mf_fy(alpha_rear, Fz_rear, tire_coefs)
    # fy_front, fy_rear = lateral_brush(Fz, C_alpha, alpha, vehicle_coefs, mu)
    # fy_front, fy_rear = linear_model(C_alpha, alpha)
    
    Izz = (1/12)*vehicle_coefs['vehicleMass']*(vehicle_coefs['trackwidth']**2 + vehicle_coefs['wheelbase']**2)
    
    beta_dot = (fy_front + fy_rear)/(vehicle_coefs['vehicleMass']*vx) - r
    r_dot = (vehicle_coefs['a']*fy_front - vehicle_coefs['b']*fy_rear)/Izz

    return r_dot

def beta_dot_eqn(X, r):
    beta = X

    Fz_front, Fz_rear = normal_forces(vehicle_coefs)
    
    alpha_front = beta + (vehicle_coefs['a']*r)/vx - delta
    alpha_rear = beta - (vehicle_coefs['b']*r)/vx
    
    Fz = (Fz_front, Fz_rear)
    C_alpha = (C_alpha_front, C_alpha_rear)
    alpha = (alpha_front, alpha_rear)
    
    fy_front = mf_fy(alpha_front, Fz_front, tire_coefs)
    fy_rear = mf_fy(alpha_rear, Fz_rear, tire_coefs)
    # fy_front, fy_rear = lateral_brush(Fz, C_alpha, alpha, vehicle_coefs, mu)
    # fy_front, fy_rear = linear_model(C_alpha, alpha)
    
    Izz = (1/12)*vehicle_coefs['vehicleMass']*(vehicle_coefs['trackwidth']**2 + vehicle_coefs['wheelbase']**2)
    
    beta_dot = (fy_front + fy_rear)/(vehicle_coefs['vehicleMass']*vx) - r
    r_dot = (vehicle_coefs['a']*fy_front - vehicle_coefs['b']*fy_rear)/Izz

    return beta_dot

"""
IMPLEMENTATION
"""

# user-specific parameters
vehicle = Vehicle("VehicleParameters.xlsx")
vehicle.load_data()
tire = TireModel("TireParameters.xlsx")
tire.load_data()

vehicle_coefs = vehicle.vehicle_params
tire_coefs = tire.tire_params

vx = 25 # m/s
delta_range = [np.deg2rad(1), np.deg2rad(5), np.deg2rad(10), np.deg2rad(15)] # steer angle
N = 20

# side slip
beta_max = 2 # rad

# yaw velocity
r_max = 2 # rad/s

# compute the values needed for the graph
def compute_graph(vehicle_coefs, tire_coefs, vx, delta, N, beta_max, r_max):

    beta_range = np.linspace(-beta_max, beta_max, N)
    r_range = np.linspace(-r_max, r_max, N)
    
    # x = beta, y = r
    beta_vec, r_vec = np.meshgrid(beta_range, r_range)
    beta_dot_values = np.zeros_like(beta_vec)
    r_dot_values = np.zeros_like(r_vec)
    for i in range(N):
        for j in range(N):
            beta = beta_vec[i, j]
            r = r_vec[i,j]
    
    
            Fz_front, Fz_rear = normal_forces(vehicle_coefs)
    
            alpha_front = beta + (vehicle_coefs['a']*r)/vx - delta
            alpha_rear = beta - (vehicle_coefs['b']*r)/vx
            
            Fz = (Fz_front, Fz_rear)
            C_alpha = (vehicle_coefs['c_alpha_f'], vehicle_coefs['c_alpha_r'])
            alpha = (alpha_front, alpha_rear)
    
            fy_front = mf_fy(alpha_front, Fz_front, tire_coefs)
            fy_rear = mf_fy(alpha_rear, Fz_rear, tire_coefs)
            # fy_front, fy_rear = lateral_brush(Fz, C_alpha, alpha, vehicle_coefs, mu)
    
            # calculate Izz based on a rectangular prism (very rough estimate, kinda really not good actually)
            Izz = (1/12)*vehicle_coefs['vehicleMass']*(vehicle_coefs['trackwidth']**2 + vehicle_coefs['wheelbase']**2)
    
            beta_dot = (fy_front + fy_rear)/(vehicle_coefs['vehicleMass']*vx) - r
            r_dot = (vehicle_coefs['a']*fy_front - vehicle_coefs['b']*fy_rear)/Izz
    
    
            magnitude = np.sqrt(beta_dot**2 + r_dot**2) + 1e-8
            beta_dot = beta_dot/magnitude
            r_dot = r_dot/magnitude
    
            beta_dot_values[i, j] = beta_dot
            r_dot_values[i, j] = r_dot


    return beta_vec, r_vec, beta_dot_values, r_dot_values

def build_plot(delta, vx, beta_vec, r_vec, beta_dot_values, r_dot_values, fixed_points=None,
               stability_values=None):
    color_list = {'stable': 'green', 'unstable': 'red', 'saddle': 'orange', 'marginal': 'yellow'}
    # build the base plot
    fig, ax = plt.subplots()
    # ax.quiver(beta_vec, r_vec, beta_dot_values, r_dot_values,
    #            angles='xy', scale_units='xy', color='gray', scale=20, alpha=0.3)
    
    ax.streamplot(beta_vec, r_vec, beta_dot_values, r_dot_values, color='blue', density=3.0, linewidth=0.5)

    # if diagram_pts != None:
    #     # we are also plotting fixed points
    #     for pt, stability in zip(diagram_pts, diagram_stability):
    #         ax[i,j].scatter(pt[0], pt[1], color=color_list[stability], s=50, alpha=1)

    ax.set_xlabel("Beta")
    ax.set_ylabel("r")

    ax.set_title(f'delta={np.rad2deg(delta)} deg, vx={vx} m/s')
    ax.grid(True)
    return fig, ax

def build_plot_sequence(n, m, delta, vx, beta_vec_list, r_vec_list, beta_dot_values_list, r_dot_values_list, fixed_points=None, stability_values=None):
    color_list = {'stable': 'lime', 'unstable': 'red', 'saddle': 'darkorange', 'marginal': 'yellow'}
    # check if delta is the range or vx is the range we are changing
    if type(delta) == list:
        # varying steering angle
        fig, ax = plt.subplots(n, m)
        i = 0
        j = 0
        for d, beta_vec, r_vec, beta_dot_values, r_dot_values, diagram_pts, diagram_stability in zip(delta, beta_vec_list, r_vec_list, beta_dot_values_list, r_dot_values_list, fixed_points, stability_values):
            # build each individual graph ax[i, j]
            ax[i, j].quiver(beta_vec, r_vec, beta_dot_values, r_dot_values,
                       angles='xy', scale_units='xy', color='gray', scale=20, alpha=0.3)
            
            ax[i,j].streamplot(beta_vec, r_vec, beta_dot_values, r_dot_values, color='blue', density=3.0, linewidth=0.5)

            if diagram_pts != None:
                # we are also plotting fixed points
                for pt, stability in zip(diagram_pts, diagram_stability):
                    ax[i,j].scatter(pt[0], pt[1], color=color_list[stability], s=50, alpha=1)

            ax[i, j].set_xlabel("Beta")
            ax[i, j].set_ylabel("r")

            ax[i,j].set_title(f'delta={np.round(np.rad2deg(d), decimals=2)} deg, vx={vx} m/s')
            ax[i,j].grid(True)

            if j < m-1:
                j += 1
            elif i < n-1:
                # move down to the next row and reset the col position
                i += 1
                j = 0
            else:
                # we are at the end of our plot
                continue
    
        # set a figure caption
        return fig, ax

    elif type(vx) == list:
        # varying velocity
        fig, ax = plt.subplots(n, m)
        i = 0
        j = 0
        for v, beta_vec, r_vec, beta_dot_values, r_dot_values in zip(vx, beta_vec_list, r_vec_list, beta_dot_values_list, r_dot_values_list):
            # build each individual graph ax[i, j]
            ax[i, j].quiver(beta_vec, r_vec, beta_dot_values, r_dot_values,
                       angles='xy', scale_units='xy', color='gray', scale=20, alpha=0.3)
            
            ax[i,j].streamplot(beta_vec, r_vec, beta_dot_values, r_dot_values, color='blue', density=3.0, linewidth=0.5)
            
            if fixed_points != None:
                # we are also plotting fixed points
                for diagram_pts, diagram_stability in zip(fixed_points, stability_values):
                    for pt, s in zip(diagram_pts, diagram_stability):
                        ax[i,j].scatter(pt[0], pt[1], color=color_list[s])


            ax[i, j].set_xlabel("Beta")
            ax[i, j].set_ylabel("r")

            ax[i,j].set_title(f'delta={np.round(np.rad2deg(delta), decimals=2)} deg, vx={v} m/s')
            ax[i,j].grid(True)

            if j < m:
                j += 1
            elif i < n:
                # move down to the next row and reset the col position
                i += 1
                j = 0
            else:
                # we are at the end of our plot
                continue


        return fig, ax
    else:
        # invalid for a plot sequence
        return None

beta_vec_list = []
r_vec_list = []
beta_dot_values_list = []
r_dot_values_list = []
fixed_pts_list = []
stability_list = []

for delta in delta_range:
    beta_vec, r_vec, beta_dot_values, r_dot_values = compute_graph(vehicle_coefs, tire_coefs, vx, delta, N, beta_max, r_max)
    fixed_pts = find_fixed_pts(beta_max, r_max)
    stability_values = []

    # classifying fixed points
    for fp in fixed_pts:
        e, s = classify_fixed_pt(fp)
        stability_values.append(s)

    beta_vec_list.append(beta_vec)
    r_vec_list.append(r_vec)
    beta_dot_values_list.append(beta_dot_values)
    r_dot_values_list.append(r_dot_values)
    fixed_pts_list.append(fixed_pts)
    stability_list.append(stability_values)
  
fig, ax = build_plot_sequence(2,2, delta_range, vx, beta_vec_list, r_vec_list, beta_dot_values_list, r_dot_values_list, fixed_pts_list, stability_list)


plt.figtext(0.5, 0.01, "stable = lime; unstable = red; saddle = orange; marginal = yellow", wrap=True, horizontalalignment="center", fontsize=12)
plt.tight_layout()
fig.show()
plt.show()

beta_vec_new = beta_vec_list[0]
r_vec_new = r_vec_list[0]
beta_dot_values_new = beta_dot_values_list[0]
r_dot_values_new = r_dot_values_list[0]

this_delta = delta_range[0]
this_vx = vx

# fig2, ax2 = build_plot(this_delta, this_vx, beta_vec_new, r_vec_new, beta_dot_values_new, r_dot_values_new)
# fig2.savefig("test_example_no_quiver.png")
# fig2.show()
# plt.show()

