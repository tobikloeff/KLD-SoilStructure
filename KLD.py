# -*- coding: utf-8 -*-


from pandas import read_csv, DataFrame
from numpy import array, unique, sqrt, pi, linspace, round_, append, log, exp, argmin
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters
from sklearn.metrics import mean_squared_error
from scipy.special import erfc
from soiltexture import getTexture
from scipy.optimize import least_squares



#%% DEFINE FUNCTIONS USED IN THE SCRIPT

def Kosugi_BM_pdf(x, par):
    """
    This is the probability density function after Kosugi (1996) for bimodal 
    pore-size distributions. Matric potential and equivalent pore radii 
    (variable x) can be used interchangeably.

    Parameters
    ----------
    x : list or array-like
        The values of matric potential/equivalent pore radius.
    par : array
        Contains the parameters of the distribution in the order:
        [effective water content (1), sigma (1), median (1), effective water
        content (2), sigma (2), median (2)].

    Returns
    -------
    pdf : array
        The values of the probability density function.

    """
    
    pdf = (par[0]/(x * par[1] * sqrt(2*pi)) * exp(-((log(x/par[2]))**2)/(2*par[1]**2))) + \
           (par[3]/(x * par[4] * sqrt(2*pi)) * exp(-((log(x/par[5]))**2)/(2*par[4]**2)))
    
    return pdf


def Kosugi_BM_integ(x, p, resid):    
    """
    This is the integrated probability density function after Kosugi (1996) 
    for bimodal pore-size distributions. Matric potential and equivalent pore 
    radii (variable x) can be used interchangeably.

    Parameters
    ----------
    x : list or array-like
        The values of matric potential/equivalent pore radius.
    p : array
        Contains the parameters of the distribution in the order:
        [effective water content (1), sigma (1), median (1), effective water
        content (2), sigma (2), median (2)].
    resid : float
            Residual water content.

    Returns
    -------
    wc : array
        The water contents.

    """
    
    wc = resid + 0.5 * (p[0] - resid) * erfc(-((log(x) - log(p[2]))/(p[1] * sqrt(2)))) + 0.5 * p[3] * erfc(-((log(x) - log(p[5]))/(p[4] * sqrt(2))))
    
    return wc


def resid_Kosugi_BM_integ(params, x, resid, observed):    
    """
    Computes the vector/array of resiudals, which is used to solve the nonlinear 
    least-squares problem (bimodal Kosugi 1996 model).

    Parameters
    ----------
    params : array
        Contains the initial parameter guesses.   
    x : list or array-like
        The values of matric potential/equivalent pore radius.
    resid : float
            Residual water content.
    observed : list or array-like
        The measured (structured soil) or derived (reference soil) water
        contents.

    Returns
    -------
    Array
        Differences between measured and calculated water contents.

    """
    
    # extract parameter guesses
    por1 = params["por1"].value
    sig1 = params["sig1"].value
    med1 = params["med1"].value
    por2 = params["por2"].value
    sig2 = params["sig2"].value
    med2 = params["med2"].value
    
    wc = resid + 0.5 * (por1 - resid) * erfc(-((log(x) - log(med1))/(sig1 * sqrt(2)))) + 0.5 * por2 * erfc(-((log(x) - log(med2))/(sig2 * sqrt(2))))

    return wc - observed


def resid_Kosugi_UM_integ(params, observed, x, resid, eff_wc=None): 
    """
    Computes the vector/array of resiudals, which is used to solve the nonlinear 
    least-squares problem (unimodal Kosugi 1996).

    Parameters
    ----------
    params : array
        Contains the initial parameter guesses.
    observed : list or array-like
        Observed values to which the nonlinear model is fitted.
    x : list or array-like
        The values of matric potential/equivalent pore radius.
    eff_wc : float
        The effective water content (i.e. the difference between the saturated 
        and residual water content).
    resid : float
            Residual water content.
    
    Returns
    -------
    Array
        Differences between measured and calculated water contents.

    """
    
    # por1 = params["por1"].value
    sig1 = params["sig1"].value
    med1 = params["med1"].value
    
    mat = resid + 0.5 * (eff_wc - resid) * erfc(-((log(x) - log(med1))/(sig1 * sqrt(2))))
    
    return observed - mat



def fit_Kosugi_UM_integ(obs, x, sig, mu, resid, est_eff_wc=False, eff_wc=0.4):
    """
    Fits the model after Kosugi (1996) to observed values of for cumulative water 
    content.

    Parameters
    ----------
    obs : list or array-like
        Observed values of cumulative water content to which the nonlinear model 
        is fitted.
    x : list or array-like
        The values of matric potential/equivalent pore radius.
    sig : float
        Initial guess of the parameter sigma.
    mu : float
        Initial guess of the median.
    est_eff_wc: bool
        If this is True, the effective water content will be treated as a
        fitting parameter. In this case the default of 0.4 is used as an initial 
        guess. Alternatively, this guess can be defined in the next parameter.
        The default is False.
    eff_wc : float
        The effective water content (i.e. the difference between the saturated 
        and residual water content). This parameter should be specified if 
        it is not estimated.
        The default is 0.4.

    Returns
    -------
    par : dict
        A dictionary including the parameters of the fitted model.

    """
    
    if est_eff_wc == True:
        p_init = array([eff_wc, sig, mu])
        p = least_squares(resid_Kosugi_UM_integ, p_init, method="lm", verbose=1, 
                      args=(obs, x, resid))["x"]
        par = {"eff_wc": p[0], "sigma": p[1], "median": p[2]}
    else:         
        p_init = array([sig, mu])    
        p = least_squares(resid_Kosugi_UM_integ, p_init, method="lm", verbose=1, 
                      args=(obs, x, resid, eff_wc))["x"] 
        par = {"eff_wc": eff_wc, "sigma": p[0], "median": p[1]}
    
    return par


def calculate_KL_divergence(f1, f2, par1, par2, a, b, n):
    '''
    Calculates the KL divergence numerically based on the composite 
    trapezoidal rule relying on the Riemann Sums.
    
    :param function f1: pore-size distribution function of structured soil
    :param function f1: pore-size distribution function of reference soil
    :param array par1: parameters (strctured soil)
    :param array par2: parameters (reference soil)
    :param float a: lower bound of the integral
    :param float b: upper bound of theintergal
    :param int n: number of trapezoids of equal width
    
    :return float: the KL divergence between a and b
    '''
    w = (b - a)/n
    result = 0 + sum([(f1(a+i*w, par1) * log(f1(a+i*w, par1)/f2(a+i*w, par2))) for i in range(1, n)]) + 0
    result *= w
    
    return result


#%% CALCULATE THE KL DIVERGENCE

# import water retention data
WR = read_csv("... .csv")

# import particle-size distribution data
PS = read_csv("... .csv")

# extract index (PS.index should be the same as WR.index)
ind = unique(PS.index)

# create dataframe for reference soil parameters
ref_params = DataFrame(columns=["por1_[-]", "sigma1_[-]", "median1_[cm]", 
                                            "por2_[-]", "sigma2_[-]", "median2_[cm]", 
                                            "RMSE_[%]"], index=ind)

# create dataframe for structured soil parameters
str_params = DataFrame(columns=["por1_[-]", "sigma1_[-]", "median1_[cm]", 
                                "por2_[-]", "sigma2_[-]", "median2_[cm]", 
                                "RMSE_[%]"], index=ind)

# create dictionary for KLD values
KLD_dic = dict()

# iterate over all samples ...
for i in ind: 
    
    # extract texture data
    x_tex = (PS.loc[i, "P_SIZE"].values)/20000 # convert to cm
    y_tex = PS.loc[i, "P_PERCENT"].values
    
    # define range of particle-sizes to >2 µm
    x_tex_lb = x_tex[x_tex >= 0.0001] 
    y_tex = y_tex[(len(x_tex) - len(x_tex_lb)):]      
    x_tex_ub = x_tex_lb[x_tex_lb <= 0.1]
    if (len(x_tex_lb) - len(x_tex_ub)) == 0:
        x_tex = x_tex_ub
    else:
        y_tex = y_tex[:(len(x_tex_lb) - len(x_tex_ub))]
        x_tex = x_tex_ub
    
    # extract water retention measurements
    if WR.loc[i, :].iloc[0, 1] >= WR.loc[i, :].iloc[1, 1]:   # check if porosity value is the largest
        por = WR.loc[i, :].iloc[0, 1]
        x_wr = 0.149/(WR.loc[i, "HEAD"].values/0.981)
        x_wr[0] = 0.149
        y_wr = WR.loc[i, "THETA"].values
    else: 
        por = WR.loc[i, :].iloc[1, 1]
        x_wr = 0.149/(WR.loc[i, "HEAD"].values/0.981)[1:]
        y_wr = WR.loc[i, "THETA"].values[1:]

    # extract clay/silt/sand contents and get texture class 
    c = y_tex[0]/100
    if c == 0:
        c = 0.000000001
    si = (y_tex[abs((x_tex - 0.003)).argmin()]/100) - c
    if si == 0:
        si = 0.000000001
    sa = 1 - si - c
    if sa == 0:
        sa = 0.000000001
    FAO = getTexture(sa*100, c*100, classification="FAO")       
    
    # determine water content at permanent wilting point (PWP)
    if WR.loc[i, "HEAD"].values[-1] < 12000:
        PWP = 0.014 + 0.572*c - 0.202*c**2      # formula derived from measured PWPs
        if PWP > WR.loc[i, "THETA"].values[-1]: # check if calculated PWP is larger than water content at lowest measured tension
            PWP = WR.loc[i, "THETA"].values[-1]
            x_wr = append(x_wr, 0.149/(15000/0.981))
            y_wr = append(y_wr, PWP)
        else:
            x_wr = append(x_wr, 0.149/(15000/0.981))
            y_wr = append(y_wr, PWP)
    else:
        PWP = WR.loc[i, "THETA"].values[-1]
        
    # extract porosity of reference soil based on FAO texture classes (values represent 1st percentiles of porosities)
    if FAO == "fine":
        por_ref = 0.37  # n = 156
    elif FAO == "medium":
        por_ref = 0.31  # n = 220
    elif FAO == "coarse":
        por_ref = 0.34  # n = 41
    
    # difference between porosity and PWP
    Diff = por_ref - PWP    
    
    # define residual water content
    resid = 0
    
    # define scaling factor between particle- and pore-size (based on Chang et al., 2019)
    scale = 0.3
    
    # translate particle into pore radius
    r_ref = array([0.149/(15000/0.981)])
    r_ref = append(r_ref, scale * array([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]))
    
    # get indices for particle size fractions and extract percentages of each fraction
    ls = []
    for f in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
        a = argmin(abs(x_tex - f))
        ls.append(a)
    y_tex = y_tex/100
    y_tex = y_tex[ls]     
    
    wc_ref = [PWP]
    for j in range(len(y_tex)-1):
        wc_ref.append((Diff * (y_tex[j+1]- y_tex[j])/(1-y_tex[0])))    
    wc_ref = array(wc_ref).cumsum()     

    # define initial parameter guesses and bounds (reference soil)
    params = Parameters()
    params.add("por1", value=0.15, min=0, max=por_ref)
    params.add("sig1", value=1, min=0, max=200)
    params.add("med1", value=0.0001, min=0, max=1)
    params.add("por2", value=por_ref - 0.15, min=0, max=por_ref)
    params.add("sig2", value=1, min=0, max=200)
    params.add("med2", value=0.001, min=0, max=1)
    
    # fit referenc soil and extract fitted parameter values (least-square method)
    result = minimize(resid_Kosugi_BM_integ, params, args=(r_ref, resid, wc_ref), method='leastsq')            
    p_ref = [result.params["por1"].value, result.params["sig1"].value, result.params["med1"].value, 
             result.params["por2"].value, result.params["sig2"].value, result.params["med2"].value]

    # determine RMSE and add fitted parameters to parameter DataFrame
    RMSE_ref = mean_squared_error(wc_ref, Kosugi_BM_integ(r_ref, p_ref, resid), squared=False)
    ref_params.loc[i, :] = [p_ref[0], p_ref[1], p_ref[2], 
                            p_ref[3], p_ref[4], p_ref[5],
                            round_(RMSE_ref, 4)]
    
    # calculate values for reference soil plot
    x_new_ref = linspace(r_ref.min(), r_ref.max(), 40000)
    y_new_ref = Kosugi_BM_integ(x_new_ref, p_ref, resid)
    
    # plot reference soil
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.scatter(wc_ref, r_ref, edgecolors='k', facecolors='none', clip_on=False)
    ax.plot(y_new_ref, x_new_ref, c='tab:red', linewidth=2, alpha=0.9)
    ax.set_title(str(i))
    ax.set_ylabel("Pore radius [cm]")
    ax.set_xlabel("Water content [cm$^{3}$ cm$^{-3}$]") 
    
    # save reference soil plot
    fig.savefig(".../" + str(i) + ".png", dpi=70)
    # plt.close()
    
    
    # estimate initial parameter guesses for structured soil (based on Klöffel et al., 2022)   
    p_tex_UM = fit_Kosugi_UM_integ(y_tex, x_tex, 1, 0.1, 0, eff_wc=100)
    med_ref = 0.816 * p_tex_UM["median"] * sqrt((0.30/(1 - 0.30)))
    sig_ref = p_tex_UM["sigma"]
       
    # define initial parameter guesses and bounds (structured soil)
    params = Parameters()
    params.add("por1", value=0.35, min=0, max=por)
    params.add("sig1", value=1.5*sig_ref, min=0, max=200)
    params.add("med1", value=1*med_ref, min=0, max=1)
    params.add("por2", value=por-0.35, min=0, max=por)
    params.add("sig2", value=0.5, min=0, max=200)
    params.add("med2", value=0.01, min=0, max=1)
            
    # fit structured soil and extract fitted parameter values (least-square method)
    result = minimize(resid_Kosugi_BM_integ, params, args=(x_wr, resid, y_wr), method='leastsq')          
    p_wr = [result.params["por1"].value, result.params["sig1"].value, result.params["med1"].value, 
            result.params["por2"].value, result.params["sig2"].value, result.params["med2"].value]
    
    # calculate values for structured soil plot
    x_new_wr = linspace(x_wr.min(), 0.149, 40000)
    y_new_wr = Kosugi_BM_integ(x_new_wr, p_wr, resid)
    
    # plot structured soil
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_yscale("log")  
    ax.scatter(y_wr, x_wr, edgecolors='k', facecolors='none', clip_on=False)
    ax.plot(y_new_wr, x_new_wr, 'tab:blue', linewidth=2)
    ax.set_title(str(i))
    ax.set_ylabel("Pore radius [cm]")
    ax.set_xlabel("Water content [cm$^{3}$ cm$^{-3}$]") 
    
    # save structured soil plot
    fig.savefig(".../" + str(i) + ".png", dpi=70)
    # plt.close()
    
    
    # Calculate KLD value numerically (trapezoidal rule)
    KLD = calculate_KL_divergence(Kosugi_BM_pdf, Kosugi_BM_pdf, p_wr, p_ref, 0.149/(15000/0.981), 0.149, 1e7)
    KLD_dic[i] = KLD
