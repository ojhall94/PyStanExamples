import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns
import sys
import corner
import ClosePlots as cp


if __name__ == "__main__":

    '''Build the data'''
    npts = 500
    true_sigma_rc = .2
    true_M_rc = -1.61
    true_sigma_out = 1.
    true_fout = .3


    RC = np.random.randn(int(npts*(1-true_fout))) * true_sigma_rc + true_M_rc
    out = np.random.randn(int(npts*(true_fout))) * true_sigma_out + true_M_rc

    d_Mi = np.append(RC, out)
    # d_Mi = RC

    y = np.random.uniform(size=d_Mi.shape)
    s = np.random.uniform(size=d_Mi.shape) *.1

    plt.errorbar(d_Mi, y, xerr = s,fmt='o')
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
