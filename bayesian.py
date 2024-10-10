import numpy as np
import emcee

def log_prior(theta):
    A, B = theta
    if (np.abs(A) > 30) or (np.abs(B) > 30):
        return -np.inf
    else:
        return 0

def log_likelihood(theta, M_vec_0, M_vec_1, O_vec, eO_vec):
    A, B = theta
    return -np.sum(np.square((
        np.log10(
            np.power(10, A+M_vec_0)+\
            np.power(10, B+M_vec_1)
        )-O_vec)/eO_vec))

def log_probability(theta, M_vec_0, M_vec_1, O_vec, eO_vec):
    lgp = log_prior(theta)
    if np.isfinite(lgp):
        lgl = log_likelihood(theta, M_vec_0, M_vec_1, O_vec, eO_vec)
        return lgl + lgp
    else:
        return -np.inf

def func_A(M_vec, O_vec, eO_vec, flag_adopt=None):
    eM_vec = np.full(M_vec.shape[-1], 0.0)
    flag_detect = (~np.isnan(O_vec)) & (np.isfinite(M_vec))
    if flag_adopt is None:
        flag_adopt = flag_detect
    else:
        flag_adopt = (flag_adopt) & (flag_detect)
    O_vec = O_vec[flag_adopt]
    # eO_vec[isotope('Pb').Z-1] /= 2
    eO_vec = eO_vec[flag_adopt]
    M_vec = M_vec[flag_adopt]
    eM_vec = eM_vec[flag_adopt]
    
    inv_err2 = 1/(np.square(eO_vec) + np.square(eM_vec))
    sum_inv_err2 = np.sum(inv_err2)
    relative_res = (M_vec - O_vec) * inv_err2
    sum_relres = np.sum(relative_res)
    A = - sum_relres / sum_inv_err2
    return A # - 0.11765639216810353

def func_AB(M_vec_0, M_vec_1, O_vec, eO_vec, flag_adopt=None):
    # eM_vec = np.full(M_vec.shape[-1], 0.0)
    # flag_detect = ~np.isnan(O_vec)
    flag_detect = (~np.isnan(O_vec)) & ((np.isfinite(M_vec_0)) & (np.isfinite(M_vec_1)))
    if flag_adopt is None:
        flag_adopt = flag_detect
    else:
        flag_adopt = (flag_adopt) & (flag_detect)
    O_vec = O_vec[flag_adopt]
    eO_vec = eO_vec[flag_adopt]
    M_vec_0 = M_vec_0[flag_adopt]
    M_vec_1 = M_vec_1[flag_adopt]
    sampler = emcee.EnsembleSampler(30, 2, log_probability, args=(M_vec_0, M_vec_1, O_vec, eO_vec))
    sampler.run_mcmc(np.random.uniform(size=(30, 2)), nsteps=3000, progress=False)
    A, B = np.median(sampler.get_chain(discard=2000).reshape(-1, 2), axis=0)
    log_prob = -sampler.get_log_prob(discard=2000)
    # argmin = np.unravel_index(np.argmin(log_prob, axis=None), log_prob.shape)
    chisqr = log_prob.min()
    # A, B = sampler.get_chain(discard=2500)[argmin]

    return A, B, chisqr# - 0.11765639216810353

def func_chisqr(A, M_vec, O_vec, eO_vec, flag_adopt=None):
    eM_vec = np.full(M_vec.shape[-1], 0.0)
    flag_detect = (~np.isnan(O_vec)) & (np.isfinite(M_vec))
    if flag_adopt is None:
        flag_adopt = flag_detect
    else:
        flag_adopt = (flag_adopt) & (flag_detect)
    AmM2_vec = np.power(A + M_vec[flag_adopt] - O_vec[flag_adopt], 2)
    eAmM2_vec = np.power(eM_vec[flag_adopt], 2) + np.power(eO_vec[flag_adopt], 2)
    chisqr = np.sum(AmM2_vec/eAmM2_vec)
    return chisqr