#!/usr/bin/env python3
from linker import *
import math

# Plot results
def Plot_distribution(particle_list_time, rp, parameters, sample=-1, save=False):
    time = particle_list_time.time.values
    numerical_sim = particle_list_time[particle_list_time.time == time[sample]].copy()
    fig, ax = plt.subplots(num=None, figsize=(9, 8),dpi=80,
                           facecolor='w', edgecolor='k')
    plt.plot(numerical_sim["zp"].values,
             numerical_sim["nz"].values/np.sum(numerical_sim["nz"].values),"-r",label="Numerical")
    plt.ylabel("Charge fraction, f (-)", fontsize=20)
    plt.xlabel("Elementary charges, z (-)", fontsize=20)
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16)
    plt.rc('mathtext', fontset='stixsans')

    y_min = 1e-04
    y_max = 0.8
    x_min = 0
    x_max = 160
    #ax.set_yscale("log");
    ax.set_ylim([y_min,y_max])
    #ax.set_xlim([x_min,x_max])
    ax.text(0.07*(x_max-x_min)+x_min,
            0.85*(y_max-y_min)+y_min,
             "$D_p$ = "+str(round(rp*2e+09,1))+" nm\n"+\
             "ESP V = "+str(int(parameters["V"]/1000))+" kV",
             color="k",fontsize=20)
    plt.show()
    if(save):
        plt.savefig('Figures/Charge_dist_simulation_v2_ESPV_'+\
                    str(int(parameters["V"]/1000))+'_Dp_'+\
                    str(int(2*rp*1e+09))+'nm.png');
    return
def Plot_in_time(particle_list_times_t,
                 parameters,
                 rp,
                 column="penetration",
                 save=False):
    fig, ax = plt.subplots(num=None, figsize=(9, 8),dpi=80,
                           facecolor='w', edgecolor='k')
    plt.plot(particle_list_times_t["time"],
             particle_list_times_t[column],
             label="simulated")
    if(column == "penetration"):
        plt.ylabel("penetration, $n_{tot}(t)/n_{tot}(0)$ (-)", fontsize=20)
    elif(column == "zp_avg"):
        plt.ylabel(r"average charge, $\bar{z}$ (-)", fontsize=20)
    elif(column == "zp_var"):
        plt.ylabel(r"charge variance, $\sigma_z^2$ (-)", fontsize=20)
    else:
        plt.ylabel(column, fontsize=20)
    plt.xlabel("residence time, $t$ (s)", fontsize=20)
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16)
    plt.rc('mathtext', fontset='stixsans')

    y_min = np.min(particle_list_times_t[column]);
    y_max = np.max(particle_list_times_t[column])
    x_min = np.min(particle_list_times_t["time"]); 
    x_max = np.max(particle_list_times_t["time"])
    ax.text(0.7*(x_max-x_min)+x_min,
             0.9*(y_max-y_min)+y_min,
             "$D_p=$"+str(round(rp*2e+09,1))+" nm\n"+\
             "ESP V = "+str(int(parameters["V"]/1000))+" kV",
             color="k",
             fontsize=20)
    plt.show()
    if(save):
        plt.savefig('Figures/Plot_in_time_'+column+'.png');
    return

# Support functions
def Statistics_time(particle_list_time):
    particle_list_time["nz_x_z"] = particle_list_time["nz"] * particle_list_time["zp"]
    particle_list_time["nz_x_z_2"] = particle_list_time["nz"] * np.power(particle_list_time["zp"],2)

    particle_list_time["nz_x_KnD"] = particle_list_time["nz"] * particle_list_time["KnD"]
    particle_list_time["nz_x_zs"] = particle_list_time["nz"] * particle_list_time["zs"]
    particle_list_times_t = particle_list_time[["nz_x_z","nz_x_z_2",
                                            "nz","nz0","time",
                                            "Rp","nz_x_KnD","nz_x_zs"]].groupby(["time","Rp"]).sum()
    particle_list_times_t.reset_index(inplace=True)
    return particle_list_times_t

### -- Particle-ion collision kernel
def Particle_ion_beta(particle,p):
    KnD = Particle_KnD(particle,p)
    beta_ip = Collision_kernel_regimes(p,particle)
    return beta_ip
def Particle_ion_beta_continuum(particle,p):
    beta_ip = Diffusive_kernel(p,particle)
    return beta_ip
def Particle_ion_beta_fm(particle,p):
    beta_ip = Ballistic_kernel(p,particle)
    return beta_ip
def Particle_ion_beta_fm2(particle,p):
    kbT = Aerosol_tools.k_B * p["T"]
    Rp = particle["Rp"]
    zp = particle["zp"]
    KE = 9.0e+09
    eta_b = np.exp(-KE*np.power(p["e"],2)*zp/(Rp*kbT))
    mi = p["mi"]
    u_i = np.sqrt(8*kbT/np.pi/mi)
    beta_b = np.pi * (Rp**2) * u_i * eta_b
    return beta_b

def Hogan_dimensionless_g(Knd,p):
    c = np.sqrt(8 * np.pi)/(4*np.pi)
    temp0 = 1 + p["c1"]*Knd/(4*np.pi) + p["c2"]*Knd**2 * c
    temp1 = 1 + p["c3"]*Knd + p["c4"]*Knd**2 + p["c2"]*Knd**3
    return temp0/temp1
def Diffusive_kernel(p,particle):
    Rp = particle["Rp"]
    zp = particle["zp"]
    fp = Aerosol_tools.friction(2*Rp, p["T"]) #particle["fp"]
    eta_c = Eta_c_only_Coulomb(p,Rp,zp)
    fi = p["fi"]
    f_ip = fi * fp / (fi + fp)
    kbT = Aerosol_tools.k_B * p["T"]
    Di = kbT/fi
    Dp = kbT/fp
    #beta_d = 4 * np.pi * (Di * Rp) * eta_c
    beta_d = 4 * np.pi * (Di+Dp) * (Rp+p["ai"]) * eta_c
    return beta_d
def Ballistic_kernel(p,particle):
    Rp = particle["Rp"]
    zp = particle["zp"]
    eta_b = Eta_b_only_Coulomb(p,Rp,zp)
    fi = p["fi"]
    mi = p["mi"]
    mp = particle["mp"]
    kbT = Aerosol_tools.k_B * p["T"]
    u_ij = np.sqrt((8*kbT/np.pi/mi) + (8*kbT/np.pi/mp))
    beta_b = np.pi * (Rp+p["ai"])**2 * u_ij * eta_b
    return beta_b

def Eta_c_only_Coulomb(p,Rp,zp):
    if (zp==0):
        return 1
    Psi_c = PSI_c(p,Rp,zp)
    return Psi_c/(1-np.exp(-Psi_c))
def Eta_b_only_Coulomb(p,Rp,zp):
    if (zp==0):
        return 1
    Psi_c = PSI_c(p,Rp,zp)
    return np.exp(Psi_c)

def Collision_kernel_regimes(p,particle):
    KnD = particle["KnD"]
    g = Hogan_dimensionless_g(KnD,p)
    beta_d = Diffusive_kernel(p,particle)
    return beta_d * g

def PSI_l(p,Rp):
    kbT = Aerosol_tools.k_B * p["T"]
    temp0 = (p["epsilon_p"]-1)/(p["epsilon_p"]-2)
    temp1 = np.power(p["e"],2)/(4*np.pi*p["epsilon_0"])
    temp2 = np.power(p["zi"],2)/(kbT * Rp)
    Psi_l =  temp0 * temp1 * temp2
    return Psi_l

def PSI_c(p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    temp1 = np.power(p["e"],2)/(4*np.pi*p["epsilon_0"])
    temp2 = 1/(kbT * Rp)
    Psi_c = - p["zi"] * zp * temp1 * temp2
    return Psi_c

def Determine_PSI_c(particle,p):
    Rp = particle["Rp"]
    zp = particle["zp"]
    return PSI_c(p,Rp,zp)

def Image_potential(r, p,Rp):
    kbT = Aerosol_tools.k_B * p["T"]
    Psi_l = PSI_l(p,Rp)
    rn = r/(Rp+p["ai"])
    U = -Psi_l * kbT/(2*rn**2*(rn**2-1))
    return U
def Coulomb_potential(r, p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    Psi_c = PSI_c(p,Rp,zp)
    rn = r/(Rp+p["ai"])
    U = -Psi_c * kbT /rn
    return U

def b_eval(r,v, p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    Phi = Image_potential(r, p,Rp) + Coulomb_potential(r, p,Rp,zp)
    b = r * np.sqrt(1+Phi/kbT/v**2)
    return b
def b_crit(v, p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    r = np.logspace(0,1,100) * (Rp+p["ai"])
    b = b_eval(r,v,p,Rp,zp) / (Rp+p["ai"])
    b = np.nan_to_num(b, nan=-1)
    try:
        b_cr = np.min(b[b>0])
    except:
        b_cr = (Rp+p["ai"])
    return b_cr
def Ouyang_fm(v,p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    b_cr = b_crit(v,p,Rp,zp)
    df = np.exp(-v**2) * v**3 * b_cr**2
    return df

def b_eval_FS(r, p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    Phi = Image_potential(r, p,Rp) + Coulomb_potential(r, p,Rp,zp)
    b = r * np.sqrt(1+2*Phi/kbT/3)
    return b
def eta_Fuchs_Sutugin_fm(p,Rp,zp):
    kbT = Aerosol_tools.k_B * p["T"]
    r = np.logspace(0,1,100) * (Rp+p["ai"])
    b = b_eval_FS(r,p,Rp,zp) / (Rp+p["ai"])
    b = np.nan_to_num(b, nan=-1)
    b_cr = np.min(b[b>0])
    return b_cr **2 * np.sqrt(3/2)

def Eta_b(p,Rp,zp):
    eta_b = 2 * quad(Ouyang_fm, 0, np.inf, args=(p,Rp,zp))[0]
    return eta_b

def New_particle(p, Rp, z):
    fp = Aerosol_tools.friction(2*Rp, p["T"])
    mu_air = Aerosol_tools.Mu_gas(p["T"])
    kbT = Aerosol_tools.k_B * p["T"]
    D = kbT/fp
    Zp = z * p["e"] / fp
    mp = p["Rho_p"] * 4*np.pi/3 * np.power(Rp,3)
    tau = mp/fp
    rho_air = Aerosol_tools.Rho_gas(p["T"])
    Re = (2*Rp) * rho_air * p["u"]/mu_air
    particle = {
        "Rp": Rp,
        "zp": z,
        "fp": fp,
        "D": D,
        "Zp": Zp,
        "mp": mp,
        "tau": tau,
        "Re": Re}
    return particle

def Particle_KnD(particle,p):
    mp = particle["mp"]
    fp = particle["fp"]
    Rp = particle["Rp"]
    zp = particle["zp"]
    eta_c = Eta_c_only_Coulomb(p,Rp,zp)
    eta_b = Eta_b_only_Coulomb(p,Rp,zp)
    mi = p["mi"]
    fi = p["fi"]
    kbT = Aerosol_tools.k_B * p["T"]
    f_ip = fi * fp / (fi + fp)
    m_ip = mi * mp / (mi + mp)
    a_ip = (Rp+p["ai"])
    #KnD = np.sqrt(kbT * mi) * eta_c/(fi*Rp*eta_b)
    KnD = np.sqrt(kbT * m_ip) * eta_c/f_ip/a_ip/eta_b
    return KnD

def Particle_eta_d(particle,p):
    Rp = particle["Rp"]
    zp = particle["zp"]
    eta_c = Eta_c_only_Coulomb(p,Rp,zp)
    return eta_c

def Particle_eta_b(particle,p):
    Rp = particle["Rp"]
    zp = particle["zp"]
    eta_b = Eta_b_only_Coulomb(p,Rp,zp)
    return eta_b

def Particle_electrophoreticV(particle,p):
    return particle["Zp"] * p["E"]

def Particle_zs(particle,p):
    temp0 = 4*np.pi*p["epsilon_0"]
    temp1 = np.power(particle["Rp"],2) * p["E"]
    temp2 = (3*p["epsilon_p"])/(p["epsilon_p"]+2)
    zs = temp0 * temp1 * temp2/p["e"]
    return np.floor(zs)

def Particle_Iz(particle,p):
    zs = Particle_zs(particle,p)
    if(particle["zp"] >= zs):
        return 0.0
    temp = p["ni"] * p["Zi"] * p["e"] * zs/(4*p["epsilon_0"])
    temp1 = np.power(1-particle["zp"]/zs,2)
    return temp * temp1

def Particle_tau_fc(particle,p):
    zs = Particle_zs(particle,p)
    k = p["e"] * p["Zi"] * p["ni"]/(4*p["epsilon_0"])
    return 1/k

def Particle_tau_d(particle,p):
    beta_ip = particle["beta_ip"]
    k = p["ni"] * beta_ip
    return 1/k

def Determine_initial_nz(particle,Dp_measured,dN_measured):
    if(particle["zp"] > 0):
        return 0.0
    N = np.interp(particle["Rp"]*2e+09,Dp_measured,dN_measured) * 1e+06
    return N

def Determine_nz_x_betaz(particle,nz):
    index = particle.name
    return nz[index] * particle["beta_ip"]

def Determine_nz_x_Iz(particle,nz):
    index = particle.name
    return nz[index] * particle["Iz"]

def Determine_dt(with_fc,with_dc, particle_list,Rp,parameters):
    dt1 = np.min(particle_list["tau_fc"][particle_list["Rp"] == Rp])/1000   # s
    dt2 = np.min(particle_list["tau_d"][particle_list["Rp"] == Rp])/18  # s
    dt3 = np.min(parameters["b"]/particle_list["v"])/100
    if(with_fc and with_dc):
        dt = np.min([dt1,dt2,dt3])
    elif(with_fc):
        dt = dt1
    elif(with_dc):
        dt = dt2
    else:
        dt = dt3
    print("    dt_fc (µs): ",dt1*1e+06," dt_dc (µs): ",dt2*1e+06," dt (µs): ",dt*1e+06)
    return dt

def nz_recursive2(beta_ip,nt,ni,t):
    nz = np.zeros_like(beta_ip)
    nz[0] = nt * np.exp(-t * beta_ip[0] * ni)
    for i in range(1,len(beta_ip)):
        prod_i = 1
        sum_i = 0
        for j in range(i):
            if(j<i):
                prod_i = prod_i * beta_ip[j] * ni
            prod_k = 1
            for k in range(i):
                if(k != j):
                    prod_k = prod_k * (beta_ip[k]-beta_ip[j]) * ni
            sum_i = sum_i + np.exp(-t * beta_ip[j] * ni)/prod_k
        nz[i] = nt * prod_i * sum_i
    nz = np.nan_to_num(nz)
    nz[nz<0] = 0
    return nz

def nz_recursive(beta_ip,nt,ni,t):
    nz = np.zeros_like(beta_ip)
    nz[0] = nt * np.exp(-beta_ip[0] * ni * t)
    for i in range(1,len(beta_ip)):
        prod_i = 1
        for j in range(i-1):
            prod_i = prod_i * beta_ip[j] * ni
         
        if(prod_i<1e-036):
            nz[i] = 0
        else:
            sum_exp_i = 0
            for j in range(i):
                prod_j = 1
                for k in range(i):
                    if(k != j):
                        prod_j = prod_j * (beta_ip[k]-beta_ip[j]) * ni
                sum_exp_i = sum_exp_i + np.exp(-beta_ip[j] * ni * t)/prod_j
            nz[i] = nt * prod_i * sum_exp_i
    nz = np.nan_to_num(nz)
    nz[nz<0] = 0
    return nz

def Simulate_Rp(Rp,
                dt,
                particle_list0,
                parameters,
                with_fc,
                with_dc,
                with_el,
                it_max = 2000000,
                sampling_each=1,
                nz_max_perc=0.02,
               recursive=False):
    if(recursive):
        beta_ip_all = particle_list0["beta_ip"].values
    particle_list = particle_list0[particle_list0["Rp"] == Rp].copy()
    particle_list.reset_index(inplace=True)
    t = 0
    v = particle_list["v"].values.copy()
    nz = particle_list["nz0"].values.copy()
    it = 0
    exported_it = 0
    particle_list["nz"] = nz
    particle_list["time"] = t
    particle_list_time = particle_list.copy()
    continue_sim = True
    
    while continue_sim:
        continue_sim = ((t< parameters["t_res"]) and (it < it_max))
        if(with_fc):
            nz_x_Iz = particle_list.apply(Determine_nz_x_Iz,axis=1,args=(nz,))
        else:
            nz_x_Iz = np.zeros_like(v)
        if(with_dc):
            nz_x_betaz = particle_list.apply(Determine_nz_x_betaz,axis=1,args=(nz,))
        else:
            nz_x_betaz = np.zeros_like(v)
    
        nzp = np.zeros_like(nz)
        if(recursive):
            nzp = nz_recursive(beta_ip_all,particle_list.nz0.iloc[0],parameters["ni"],t)
        else:
            nzp[0] = nz[0] - dt * nz[0] * v[0]/parameters["b"] - dt * parameters["ni"] * nz_x_betaz[0]-\
                 dt * nz_x_Iz[0]
            for i in range(1,len(nz)):
                dnz_diffusion = - dt*parameters["ni"]*nz_x_betaz[i] +\
                              dt*parameters["ni"]*nz_x_betaz[i-1]
                dnz_fieldCharg = - dt*nz_x_Iz[i] +\
                               dt*nz_x_Iz[i-1]
                dnz_el = 0
                if(with_el):
                    dnz_el = - dt * nz[i] * v[i]/parameters["b"]
                nzp[i] = nz[i] + dnz_diffusion + dnz_fieldCharg + dnz_el
        nz = nzp.copy()
        # Stop the simulation when error is found
        if(nz[-1]/np.max(nz) > nz_max_perc):
            print("Simulation ERROR: z_max attained the maximum level.")
            continue_sim = False
        continue_sim = (continue_sim and np.sum(nz)/np.sum(particle_list["nz0"].values)>1e-02)
        it += 1
        t += dt
        if(math.floor(t/sampling_each) >= exported_it):
            particle_list["nz"] = nz
            particle_list["time"] = t
            particle_list_time = pd.concat([particle_list_time,particle_list])
            #print("t/t_res ", t/parameters["t_res"], " t ", t)
            exported_it = exported_it + 1
    particle_list["nz"] = nz
    particle_list["time"] = t
    if(particle_list_time.time.values[-1] != t):
        particle_list_time = pd.concat([particle_list_time,particle_list])
    del nz,nzp
    return particle_list_time



