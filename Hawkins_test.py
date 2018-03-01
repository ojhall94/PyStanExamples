import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns
import sys
import corner
import ClosePlots as cp


if __name__ == "__main__":

    '''Build the data'''
    npts = 5000
    true_sigma_rc = .2
    true_M_rc = -1.61
    true_sigma_out = 1.
    true_fout = .3

    RC = np.random.randn(int(npts*(1-true_fout))) * true_sigma_rc + true_M_rc
    out = np.random.randn(int(npts*(true_fout))) * true_sigma_out + true_M_rc

    d_Mi = np.append(RC, out)
    # d_Mi = RC

    '''Now lets build all the other components, namely distance.'''
    d_m0 = np.random.randn(len(d_Mi)) * 1. + 12.5     #Build some distance moduli
    d_ri = 10 ** (d_m0/5 + 1)                       #Build some distances
    d_wi = 1000/d_ri                                #Build some parallaxes IN MAS

    '''Av takes a little more care'''
    avleft = np.random.randn(int(len(d_Mi)*10)) * 0.04 + 0.2    #Building an assymmetric gaussian dist
    avright = np.random.randn(int(len(d_Mi)*10)) * 0.3 + 0.2    #Note: these have too many points, needed for asymmetry weighting
    d_Av = np.append(avleft[avleft<=0.2][0:800],avright[avright>0.2][0:4200])   #Combine the two components
    plt.show()

    '''Finally put all the pieces together for mi'''
    d_Aks = 0.306 * d_Av #Yuan, Liu & Xiang 2013
    d_mi = d_Mi + 5*np.log10(d_ri) - 5 + d_Aks     #Build apparent mags without reddening

    '''Now lets build the errors'''
    e_wi = np.random.randn(len(d_Mi))*0.1 + (0.18*d_wi)   #TGAS error in mas
    e_mi = np.ones_like(d_mi) * 0.02       #Magnitude error on Ks


    plt.scatter(d_Mi, d_mi, s=3, c=d_Aks)
    plt.colorbar()
    # plt.errorbar(d_Mi, d_mi, xerr = s,fmt='o')
    plt.show()

    plt.errorbar(d_wi,d_mi, xerr=e_wi, yerr=e_mi,fmt=",k",alpha=.1, ms=0, capsize=0, lw=1, zorder=999)
    plt.scatter(d_wi,d_mi,s=2,zorder=1001)
    plt.show()

    '''Build the model'''
    rc_code_1 = '''
        data{
            int<lower=0> j;
            real dMi[j];
            real<lower=0> s[j];
        }
        parameters {
            real mu;
            real<lower = 0, upper=.1> sigma;

            real eta[j];
        }
        transformed parameters {
            real theta[j];
            for (i in 1:j)
            theta[i] = mu + sigma * eta[i];
        }
        model{
            eta ~ normal(0, 1);
            dMi ~ normal(theta, s);
        }
    '''

    rc_code_2 = '''
        data{
            int<lower=0> j;
            real dMi[j];
            real<lower=0> s[j];
        }
        parameters {
            real mu;
            real<lower = 0> sigmarc;
            real<lower = 0> sigmaout;
            real<lower=0,upper=1> fout;
            real eta[j];
        }
        transformed parameters {
            real theta[j];

            for (i in 1:j)
                theta[i] = (mu + sigmarc*eta[i]);
        }
        model{
            eta ~ normal(0, 1);
            dMi ~ normal(theta, s);
        }
    '''
    sm = pystan.StanModel(model_code=rc_code_2, model_name='Simple_RC')

    dat = {'j': len(d_Mi),
            'dMi' : d_Mi,
            's' : s}

    fit = sm.sampling(data=dat, iter=2000, chains=4)
    fit.plot()
    plt.tight_layout()

    samples = fit.extract(permuted=True)
    # samples = np.array([samples['mu'],samples['sigma']]).T
    # corner.corner(samples, labels=['mu','sigma'], truths=[true_M_rc, true_sigma_rc])
    samples = np.array([samples['mu'],samples['sigmarc'],samples['sigmaout']]).T
    corner.corner(samples, labels=['mu','sigmarc','sigmaout','fout'], truths=[true_M_rc, true_sigma_rc, true_sigma_out, true_fout])



    cp.show()
    sys.exit()
