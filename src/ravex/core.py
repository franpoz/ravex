import numpy as np
import scipy.optimize as sp
from scipy.optimize import curve_fit
import astropy.constants as c
import astropy.units as u
import astropy.time as t
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
import time
import re
import warnings
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os


################################
# === helpers for statistics ===
################################

def safe_sigma_from_fap(
    FAP,
    *,
    zero_if_bad=True,
    floor=1e-300,
    ceil=1-1e-16,
    cap_sigma=None
):
    """
    Convert FAP -> sigma_equiv (norm.isf(FAP)) with safeguards:
      - invalid FAP or values outside (0,1): σ=0
      - FAP >= 0.5 -> σ=0 (it does not make sense as a 'detection')
      - very small FAP -> clipped by 'floor' to avoid +inf
      - cap_sigma: optional upper limit (e.g., 12.0)
    """
    import numpy as np
    from scipy.stats import norm

    if (FAP is None) or (not np.isfinite(FAP)):
        return 0.0 if zero_if_bad else np.nan
    if FAP <= 0.0:
        FAP = floor
    elif FAP >= 1.0:
        return 0.0 if zero_if_bad else np.nan
    if FAP >= 0.5:
        return 0.0

    FAP = np.clip(float(FAP), floor, ceil)
    sigma = float(norm.isf(FAP))  # 1-cola
    if (not np.isfinite(sigma)) or (sigma < 0.0):
        return 0.0
    if (cap_sigma is not None) and np.isfinite(cap_sigma):
        sigma = min(sigma, float(cap_sigma))
    return sigma


def _bootstrap_fap_maxpower(ls, jd, rv, dy, freq, p_obs, n_bootstrap, rng):
    """
    Estimate the FAP by bootstrap of the 'maximum power' on the same frequency grid:
    - Shuffle rv while keeping jd and dy fixed.
    - Compute power(freq) and take its maximum.
    - FAP ≈ Pr[max(power_bootstrap) ≥ p_obs].
    Add-one smoothing is used to avoid exact 0.
    """
    import numpy as np
    from astropy.timeseries import LombScargle

    exceed = 0
    for _ in range(int(n_bootstrap)):
        rv_shuf = rng.permutation(rv)
        ls_b = LombScargle(jd, rv_shuf, dy)
        p_b = ls_b.power(freq)
        if np.nanmax(p_b) >= p_obs:
            exceed += 1
    # add-one smoothing to avoid exact 0 and 1 values
    return (exceed + 1) / (int(n_bootstrap) + 1)


###################################
# === helpers instrumental ===
###################################

def equatorial_velocity(prot_days, rstar_rsun):
    """
    Estimate the stellar equatorial rotational velocity.

    Parameters
    ----------
    prot_days : float
        Stellar rotation period in days.
    rstar_rsun : float
        Stellar radius in solar radii.

    Returns
    -------
    v_eq_kms : float
        Equatorial rotational velocity in km/s.
    """
    try:
        prot_days = float(prot_days)
        rstar_rsun = float(rstar_rsun)
    except (TypeError, ValueError):
        raise TypeError("prot_days and rstar_rsun must be numeric.")

    if prot_days <= 0:
        raise ValueError("prot_days must be > 0.")
    if rstar_rsun <= 0:
        raise ValueError("rstar_rsun must be > 0.")

    rsun_m = 6.957e8
    prot_s = prot_days * 86400.0
    rstar_m = rstar_rsun * rsun_m

    v_eq_ms = 2.0 * np.pi * rstar_m / prot_s
    v_eq_kms = v_eq_ms / 1000.0
    return v_eq_kms


def carm_error(
    jmag,
    spt,
    vsini=None,
    prot_days=None,
    rstar_rsun=None,
    transiting=False,
    assume_spin_orbit_alignment=False,
    return_diagnostics=False,
):
    """
    Estimate the average CARMENES-VIS RV error for an M dwarf.

    Parameters
    ----------
    jmag : float
        J-band magnitude of the target.
    spt : float
        Spectral subtype encoded numerically:
        M0 -> 0, M1 -> 1, ..., M4 -> 4, etc.
    vsini : float or None, optional
        Projected rotational velocity in km/s.
        If provided, this value is used directly.
    prot_days : float or None, optional
        Stellar rotation period in days.
        Used only if vsini is not provided.
    rstar_rsun : float or None, optional
        Stellar radius in solar radii.
        Used only if vsini is not provided.
    transiting : bool, optional
        Whether the system hosts transiting planets.
    assume_spin_orbit_alignment : bool, optional
        If True, and if transiting=True, then assume sin(i_star) ~ 1 and use
        vsini ~ v_eq.
    return_diagnostics : bool, optional
        If True, return a dictionary with intermediate values.

    Returns
    -------
    e_err : float
        Estimated RV error in m/s.

    or, if return_diagnostics=True

    info : dict
        Dictionary with:
        - e_err
        - e_base
        - snr
        - vsini_used
        - v_eq (if computed)
        - rot_factor
        - vsini_source

    Notes
    -----
    This function implements an empirical prescription with:
    - a spectral-type-dependent SNR model from J magnitude,
    - a spectral-type-dependent base RV precision,
    - and, for faster rotators, an additional rotational degradation term.

    Important:
    If vsini is not measured, using vsini ~ v_eq is an approximation that assumes
    the stellar spin axis is seen nearly edge-on. This may be reasonable for
    transiting systems if spin-orbit alignment is assumed, but it is not guaranteed.
    """

    # -------------------
    # Input validation
    # -------------------
    try:
        jmag = float(jmag)
        spt = float(spt)
    except (TypeError, ValueError):
        raise TypeError("jmag and spt must be numeric.")

    if vsini is not None:
        try:
            vsini = float(vsini)
        except (TypeError, ValueError):
            raise TypeError("vsini must be numeric if provided.")
        if vsini < 0:
            raise ValueError("vsini must be >= 0 km/s.")

    # -------------------
    # Infer vsini if needed
    # -------------------
    v_eq = None
    vsini_source = None

    if vsini is None:
        if prot_days is None or rstar_rsun is None:
            raise ValueError(
                "vsini was not provided. To infer it, you must provide both "
                "prot_days and rstar_rsun."
            )

        v_eq = equatorial_velocity(prot_days, rstar_rsun)

        if transiting and assume_spin_orbit_alignment:
            vsini = v_eq
            vsini_source = "inferred_from_veq_assuming_alignment"
        else:
            raise ValueError(
                "vsini was not provided. I can compute v_eq from prot_days and "
                "rstar_rsun, but I will only approximate vsini ~ v_eq if "
                "transiting=True and assume_spin_orbit_alignment=True."
            )
    else:
        vsini_source = "provided_directly"

    # -------------------
    # Spectral-type bin
    # -------------------
    if spt < 1:
        snr_a, snr_b = 259.65, 20.162
        base_A, base_alpha, base_C = 170.657, -1.031, 0.245
        rot_A, rot_alpha, rot_C = 0.05, 2.36, 0.897

    elif 1 <= spt < 3:
        snr_a, snr_b = 278.58, 23.986
        base_A, base_alpha, base_C = 16898.14, -2.386, 0.968
        rot_A, rot_alpha, rot_C = 0.05, 2.36, 0.897

    elif 3 <= spt < 4:
        snr_a, snr_b = 278.58, 23.986
        base_A, base_alpha, base_C = 4789.779, -2.065, 0.703
        rot_A, rot_alpha, rot_C = 0.1065, 1.76, 0.651

    else:  # spt >= 4
        snr_a, snr_b = 278.58, 23.986
        base_A, base_alpha, base_C = 25.849, -0.592, -0.708
        rot_A, rot_alpha, rot_C = 0.1065, 1.76, 0.651

    # -------------------
    # SNR estimate
    # -------------------
    snr = snr_a - snr_b * jmag

    if snr <= 0:
        raise ValueError(
            f"Estimated SNR is non-physical (snr={snr:.3f}) for jmag={jmag:.3f}. "
            "Target is outside the valid regime of this empirical relation."
        )

    # -------------------
    # Base RV error
    # -------------------
    e_base = base_A * snr**base_alpha + base_C

    # -------------------
    # Rotational penalty
    # -------------------
    if vsini <= 2:
        rot_factor = 1.0
        e_err = e_base
    else:
        rot_factor = rot_A * vsini**rot_alpha + rot_C
        e_err = e_base * rot_factor

    if return_diagnostics:
        return {
            "e_err": e_err,
            "e_base": e_base,
            "snr": snr,
            "vsini_used": vsini,
            "v_eq": v_eq,
            "rot_factor": rot_factor,
            "vsini_source": vsini_source,
        }

    return e_err


def _parse_spectral_type_maroonx(spectral_type):
    """
    Parse spectral type for MAROON-X SERVAL error relations.

    Accepted examples
    -----------------
    - "G2V", "G3", "G"
    - "K0IV-V", "K3V", "K"
    - "M3.5V", "M2", "M"
    - 2.5   -> interpreted as M2.5

    Returns
    -------
    family : str
        One of {"G", "K", "M"}.
    subtype : float or None
        Numeric subtype if available, else None.
    """
    if isinstance(spectral_type, (int, float, np.integer, np.floating)):
        return "M", float(spectral_type)

    if not isinstance(spectral_type, str):
        raise TypeError("spectral_type must be a string or a numeric M subtype.")

    s = spectral_type.strip().upper()

    m = re.match(r"^\s*([GKM])\s*([0-9]+(?:\.[0-9]+)?)?", s)
    if m is None:
        raise ValueError(
            "Could not parse spectral_type. Examples: 'G2V', 'K3V', 'M3.5V', or 2.5 for M2.5."
        )

    family = m.group(1)
    subtype = float(m.group(2)) if m.group(2) is not None else None
    return family, subtype


def _maroonx_powerlaw_sigma(snr_peak, A, alpha, C):
    """
    Generic power-law relation:
        sigma_RV = A / (SNR^alpha) + C
    """
    return A / (snr_peak ** alpha) + C


def maroonx_serval_error(snr_peak, spectral_type, arm="red", m_mode="interpolate",
                         return_diagnostics=False):
    """
    Estimate the intrinsic SERVAL RV uncertainty for MAROON-X from peak SNR.

    Parameters
    ----------
    snr_peak : float
        Peak SNR from the MAROON-X ETC for the selected arm: https://maroon-x-etc.gemini.edu/app
    spectral_type : str or float
        Spectral type of the target.

        Accepted examples:
        - G/K stars: "G2V", "G3V", "K0IV-V", "K3V", "G", "K"
        - M stars: "M0V", "M3.5V", "M6V", "M", or numeric subtype (e.g. 2.5 for M2.5)

    arm : {"blue", "red"}, optional
        MAROON-X arm for which the uncertainty is requested.

    m_mode : {"interpolate", "mean"}, optional
        Strategy for M stars:
        - "interpolate": interpolate between early-M and late-M relations using subtype
          (M0 -> early-M, M6 -> late-M). If subtype is unknown, falls back to the mean.
        - "mean": always return the mean of the early-M and late-M relations.

    return_diagnostics : bool, optional
        If True, return a dictionary with intermediate values.

    Returns
    -------
    sigma_rv : float
        Estimated SERVAL RV uncertainty in m/s for the chosen arm.

    or, if return_diagnostics=True

    info : dict
        Dictionary containing:
        - sigma_rv
        - arm
        - family
        - subtype
        - method
        - sigma_early_M (for M stars)
        - sigma_late_M (for M stars)

    Notes
    -----
    Relations implemented from the empirical fits in provided by https://www.gemini.edu/instrumentation/maroon-x/exposure-time-estimation.

    G stars:
        Blue: sigma = 735 / SNR^1.37 + 0.18
        Red : sigma = 1152 / SNR^1.27 + 0.33

    Early K stars:
        Blue: sigma = 176 / SNR^1.14 + 0.07
        Red : sigma = 416 / SNR^1.15 + 0.13

    M stars, early-M branch:
        Blue: sigma = 142 / SNR^1.00 - 0.14
        Red : sigma = 414 / SNR^1.14 + 0.01

    M stars, late-M branch:
        Blue: sigma =  99 / SNR^1.09 + 0.04
        Red : sigma = 180 / SNR^1.15 + 0.06
    """
    # -------------------
    # Input validation
    # -------------------
    if 1500 <= snr_peak < 1900:
        warnings.warn(
            "The input peak SNR is approaching the approximate MAROON-X ETC exposure "
            "limit (~1900 per 1D pixel). Please verify the exposure setup directly "
            "with the MAROON-X ETC."
        )

    elif snr_peak >= 1900:
        warnings.warn(
            "The input peak SNR is at or above the approximate MAROON-X ETC exposure "
            "limit (~1900 per 1D pixel), where detector non-linearity or saturation "
            "may become relevant. Please verify the exposure setup directly with the "
            "MAROON-X ETC."
        )

    try:
        snr_peak = float(snr_peak)
    except (TypeError, ValueError):
        raise TypeError("snr_peak must be numeric.")

    if snr_peak <= 0:
        raise ValueError("snr_peak must be > 0.")

    arm = arm.strip().lower()
    if arm not in {"blue", "red"}:
        raise ValueError("arm must be either 'blue' or 'red'.")

    m_mode = m_mode.strip().lower()
    if m_mode not in {"interpolate", "mean"}:
        raise ValueError("m_mode must be either 'interpolate' or 'mean'.")

    family, subtype = _parse_spectral_type_maroonx(spectral_type)

    # -------------------
    # Coefficients
    # -------------------
    coeffs = {
        "G": {
            "blue": (735.0, 1.37, 0.18),
            "red":  (1152.0, 1.27, 0.33),
        },
        "K": {
            "blue": (176.0, 1.14, 0.07),
            "red":  (416.0, 1.15, 0.13),
        },
        "M_early": {
            "blue": (142.0, 1.00, -0.14),
            "red":  (414.0, 1.14, 0.01),
        },
        "M_late": {
            "blue": (99.0, 1.09, 0.04),
            "red":  (180.0, 1.15, 0.06),
        },
    }

    # -------------------
    # G and K stars
    # -------------------
    if family in {"G", "K"}:
        A, alpha, C = coeffs[family][arm]
        sigma_rv = _maroonx_powerlaw_sigma(snr_peak, A, alpha, C)

        if return_diagnostics:
            return {
                "sigma_rv": sigma_rv,
                "arm": arm,
                "family": family,
                "subtype": subtype,
                "method": f"{family}-relation",
            }
        return sigma_rv

    # -------------------
    # M stars
    # -------------------
    A_e, alpha_e, C_e = coeffs["M_early"][arm]
    A_l, alpha_l, C_l = coeffs["M_late"][arm]

    sigma_early = _maroonx_powerlaw_sigma(snr_peak, A_e, alpha_e, C_e)
    sigma_late = _maroonx_powerlaw_sigma(snr_peak, A_l, alpha_l, C_l)

    if m_mode == "mean":
        sigma_rv = 0.5 * (sigma_early + sigma_late)
        method = "mean_early_late_M"

    else:  # interpolate
        if subtype is None:
            sigma_rv = 0.5 * (sigma_early + sigma_late)
            method = "mean_early_late_M_no_subtype"
        else:
            # Interpolate from M0 to M6
            w = np.clip(subtype / 6.0, 0.0, 1.0)
            sigma_rv = (1.0 - w) * sigma_early + w * sigma_late
            method = f"interpolated_M{subtype:g}"

    if return_diagnostics:
        return {
            "sigma_rv": sigma_rv,
            "arm": arm,
            "family": family,
            "subtype": subtype,
            "method": method,
            "sigma_early_M": sigma_early,
            "sigma_late_M": sigma_late,
        }

    return sigma_rv




# Globals used by the parallel detectability workers
_DETMAP_SYSTEM = None
_DETMAP_JD = None
_DETMAP_RV = None
_DETMAP_RV_ERR = None
_DETMAP_COMMON = None

# Globals used by the parallel precision-tracker workers
_PRECISION_SYSTEM = None
_PRECISION_COMMON = None


def _init_detectability_map_worker(system, jd, rv, rv_err, common_kwargs):
    """Initializer for parallel detectability-map workers."""
    global _DETMAP_SYSTEM, _DETMAP_JD, _DETMAP_RV, _DETMAP_RV_ERR, _DETMAP_COMMON
    _DETMAP_SYSTEM = system
    _DETMAP_JD = np.asarray(jd, dtype=float)
    _DETMAP_RV = np.asarray(rv, dtype=float)
    _DETMAP_RV_ERR = np.asarray(rv_err, dtype=float)
    _DETMAP_COMMON = dict(common_kwargs)


def _detectability_map_cell_worker(payload):
    """
    Worker that evaluates one (mass, period) cell of the detectability map.
    """
    system = _DETMAP_SYSTEM
    jd = _DETMAP_JD
    rv = _DETMAP_RV
    rv_err = _DETMAP_RV_ERR
    cfg = _DETMAP_COMMON

    i_m = int(payload['i_m'])
    j_p = int(payload['j_p'])
    msini = float(payload['msini'])
    period_days = float(payload['period_days'])
    phases = np.asarray(payload['phases'], dtype=float)
    rec_seeds = payload['rec_seeds']
    return_trial_details = bool(payload['return_trial_details'])

    recovered_flags = []
    hit_flags = []
    fap_vals = []
    sigma_vals = []
    k_relerr_vals = []
    p_best_vals = []
    trial_details = [] if return_trial_details else None

    for phase, rec_seed in zip(phases, rec_seeds):
        rv_inj, inj_info = system.inject_planet_in_series(
            jd,
            msini,
            period_days,
            phase=float(phase),
            eccentricity=float(cfg['eccentricity']),
            argument_periapse=cfg['argument_periapse'],
            return_metadata=True,
        )
        rv_trial = rv + rv_inj

        rec = system.recover_periodic_signal(
            jd,
            rv_trial,
            rv_err,
            min_period=float(cfg['min_period_search']),
            max_period=float(cfg['max_period_search']),
            samples_per_peak=int(cfg['samples_per_peak']),
            nyquist_factor=float(cfg['nyquist_factor']),
            fap_method=cfg['fap_method'],
            n_bootstrap=int(cfg['n_bootstrap']),
            rng_seed=int(rec_seed),
        )

        rel_period_err = abs(rec['P_best_days'] - period_days) / float(period_days)
        hit_period = rel_period_err <= float(cfg['period_tol'])

        fap_ok = rec['FAP'] < float(cfg['fap_alpha'])
        if inj_info['K_inj_mps'] > 0:
            k_relerr = abs(rec['K_best_mps'] - inj_info['K_inj_mps']) / inj_info['K_inj_mps']
        else:
            k_relerr = np.nan
        k_ok = np.isfinite(k_relerr) and (k_relerr <= float(cfg['k_tol']))

        if cfg['criterion'] == 'period_only':
            recovered = hit_period
        elif cfg['criterion'] == 'period+fap':
            recovered = hit_period and fap_ok
        elif cfg['criterion'] == 'period+k':
            recovered = hit_period and k_ok
        else:  # period+fap+k
            recovered = hit_period and fap_ok and k_ok

        recovered_flags.append(bool(recovered))
        hit_flags.append(bool(hit_period))
        fap_vals.append(float(rec['FAP']))
        sigma_vals.append(float(rec['sigma_equiv']))
        k_relerr_vals.append(float(k_relerr) if np.isfinite(k_relerr) else np.nan)
        p_best_vals.append(float(rec['P_best_days']))

        if return_trial_details:
            trial_details.append({
                'msini_mearth': float(msini),
                'period_days': float(period_days),
                'phase': float(phase),
                'P_best_days': float(rec['P_best_days']),
                'FAP': float(rec['FAP']),
                'sigma_equiv': float(rec['sigma_equiv']),
                'K_inj_mps': float(inj_info['K_inj_mps']),
                'K_best_mps': float(rec['K_best_mps']),
                'period_hit': bool(hit_period),
                'recovered': bool(recovered),
            })

    return {
        'i_m': i_m,
        'j_p': j_p,
        'recovery_rate': np.mean(recovered_flags) if recovered_flags else np.nan,
        'period_hit_rate': np.mean(hit_flags) if hit_flags else np.nan,
        'median_fap': np.nanmedian(fap_vals) if len(fap_vals) else np.nan,
        'median_sigma': np.nanmedian(sigma_vals) if len(sigma_vals) else np.nan,
        'median_k_relerr': np.nanmedian(k_relerr_vals) if len(k_relerr_vals) else np.nan,
        'median_p_best_days': np.nanmedian(p_best_vals) if len(p_best_vals) else np.nan,
        'trial_details': trial_details,
    }


def _init_precision_tracker_worker(system, common_kwargs):
    """Initializer for parallel precision-tracker workers."""
    global _PRECISION_SYSTEM, _PRECISION_COMMON
    _PRECISION_SYSTEM = system
    _PRECISION_COMMON = dict(common_kwargs)


def _precision_tracker_trial_worker(payload):
    """Worker that evaluates one Monte Carlo trial of ``precision_tracker``."""
    system = _PRECISION_SYSTEM
    cfg = _PRECISION_COMMON

    i_n = int(payload['i_n'])
    N = int(payload['N'])
    seed = int(payload['seed'])

    rng = np.random.default_rng(seed)
    start_time = t.Time(float(cfg['start_time_jd']), format='jd', scale='utc')
    sigma_eff = float(cfg['sigma_eff_mps']) * u.m / u.s
    sigma_eff_mps = float(cfg['sigma_eff_mps'])

    try:
        obs_dates = system.obs_dates(N, float(cfg['span_days']), start_time, rng=rng)
        _, _, _, phased_obs = system.get_rvs(obs_dates, noise=sigma_eff, rng=rng)

        phase_obs = phased_obs[f"p{cfg['planet_index']}"]['phase']
        y_obs = phased_obs[f"p{cfg['planet_index']}"]['rv']
        yerr_vec = np.full_like(y_obs, sigma_eff_mps, dtype=float)

        dense_dates = system.obs_dates(int(cfg['n_dense']), float(cfg['span_days']), start_time, rng=rng)

        fit_mode = cfg['fit_mode']
        p0 = cfg['initial_guess']
        bnds = cfg['bounds']
        planet_index = int(cfg['planet_index'])

        if fit_mode == "mass_gamma":
            def fmodel(phase, mass, gamma):
                return system.model_for_fit(
                    phase, mass, 0.0, dense_dates, planet_index=planet_index, gamma=gamma
                )
        elif fit_mode == "mass_e":
            def fmodel(phase, mass, e):
                return system.model_for_fit(
                    phase, mass, e, dense_dates, planet_index=planet_index, gamma=0.0
                )
        elif fit_mode == "mass_only":
            def fmodel(phase, mass):
                return system.model_for_fit(
                    phase, mass, 0.0, dense_dates, planet_index=planet_index, gamma=0.0
                )
        else:
            raise ValueError("fit_mode must be 'mass_gamma', 'mass_e', or 'mass_only'.")

        popt, pcov = curve_fit(
            fmodel,
            phase_obs,
            y_obs,
            p0=p0,
            bounds=bnds,
            sigma=yerr_vec,
            absolute_sigma=True,
            maxfev=20000,
        )

        m_fit = float(popt[0])
        if m_fit <= 0 or (not np.isfinite(m_fit)):
            return {'i_n': i_n, 'mass': np.nan, 'precision': np.nan, 'ok': False, 'error': 'non_positive_mass'}

        pcov = np.asarray(pcov, dtype=float)
        if pcov.ndim != 2 or pcov.shape[0] == 0:
            return {'i_n': i_n, 'mass': np.nan, 'precision': np.nan, 'ok': False, 'error': 'invalid_pcov'}

        var_m = float(pcov[0, 0])
        if (not np.isfinite(var_m)) or (var_m < 0):
            return {'i_n': i_n, 'mass': np.nan, 'precision': np.nan, 'ok': False, 'error': 'invalid_mass_variance'}

        dm = float(np.sqrt(var_m))
        precision = 100.0 * dm / m_fit
        if not np.isfinite(precision):
            return {'i_n': i_n, 'mass': np.nan, 'precision': np.nan, 'ok': False, 'error': 'invalid_precision'}

        return {'i_n': i_n, 'mass': m_fit, 'precision': precision, 'ok': True, 'error': None}

    except Exception as e:
        return {'i_n': i_n, 'mass': np.nan, 'precision': np.nan, 'ok': False, 'error': f"{type(e).__name__}: {e}"}



class MultiPlanetSystem(object):
    def __init__(self, mass_main, planets, use_bjd=False, location=None, target=None):
        """
        Parameters
        ----------
        mass_main : `astropy.Quantity`
            Mass of the main object of which the radial velocity is desired.
        
        planets : list of dict
            List of dictionaries containing the parameters for each planet. Each dictionary should have:
            - 'mass': `astropy.Quantity`
            - 'time_periastron': `astropy.Time`, optional
            - 'time_conjunction': `astropy.Time`, optional
            - 'inclination': `astropy.Quantity`
            - 'argument_periapse': `astropy.Quantity`
            - 'eccentricity': float
            - 'orbital_period': `astropy.Quantity` or 'semi_major_axis': `astropy.Quantity`
            - 'mean_longitude': `astropy.Quantity`, optional
            - 'reference_velocity': `astropy.Quantity`, optional
        use_bjd : bool, optional
            Whether to use Barycentric Julian Date (BJD) instead of Julian Date (JD). Default is False.
        location : `astropy.coordinates.EarthLocation`, optional
            Location of the observatory or telescope on Earth. Required if use_bjd is True.
        target : `astropy.coordinates.SkyCoord`, optional
            Coordinates of the target star. Required if use_bjd is True.
        """
        self.mass_main = mass_main
        self.use_bjd = use_bjd
        self.location = location
        self.target = target
        self.planets = [self._init_planet(p) for p in planets]

        if self.use_bjd and (self.location is None or self.target is None):
            raise ValueError('Location and target must be provided if use_bjd is True')

    def convert_to_bjd(self, time_jd):
        if not self.use_bjd:
            return time_jd
        times = t.Time(time_jd, format='jd', scale='utc', location=self.location)
        ltt_bary = times.light_travel_time(self.target)
        time_bjd = times.tdb + ltt_bary
        return time_bjd.jd

    def _init_planet(self, p):
        planet = {}
        planet['mass'] = p['mass']
        planet['time_periastron'] = p.get('time_periastron', None)
        planet['time_conjunction'] = p.get('time_conjunction', None)
        planet['inclination'] = p['inclination']
        planet['argument_periapse'] = p['argument_periapse']
        planet['eccentricity'] = p['eccentricity']
        planet['mean_longitude'] = p.get('mean_longitude', 0 * u.deg)
        planet['reference_velocity'] = p.get('reference_velocity', 0 * u.m / u.s)

        if 'orbital_period' in p:
            orbital_period = p['orbital_period']
            semi_major_axis = (orbital_period ** 2 * c.G *
                               (self.mass_main + planet['mass']) / 4 / np.pi ** 2) ** (1 / 3)
        elif 'semi_major_axis' in p:
            semi_major_axis = p['semi_major_axis']
            orbital_period = (semi_major_axis ** 3 * 4 * np.pi ** 2 /
                              c.G / (self.mass_main + planet['mass'])) ** 0.5
        else:
            raise ValueError('Either the orbital period or the semi-major axis has to be provided.')

        planet['orbital_period'] = orbital_period.to(u.d)
        planet['semi_major_axis'] = semi_major_axis.to(u.au)

        planet['_m'] = planet['mass'].to(u.solMass).value
        planet['_i'] = planet['inclination'].to(u.rad).value
        planet['_omega'] = planet['argument_periapse'].to(u.rad).value
        planet['_lambda'] = planet['mean_longitude'].to(u.rad).value
        planet['_period'] = planet['orbital_period'].to(u.d).value
        planet['_a'] = planet['semi_major_axis'].to(u.au).value
        planet['_gamma'] = planet['reference_velocity'].to(u.m / u.s).value
        planet['_grav'] = c.G.to(u.au ** 3 / u.solMass / u.d ** 2).value

        if planet['eccentricity'] >= 1:
            raise ValueError('Keplerian orbits are ellipses, therefore eccentricity must satisfy ecc < 1')

        # --- Robust handling of t0 ---
        t_peri = planet['time_periastron']
        t_conj = planet['time_conjunction']

        if t_peri is None and t_conj is None:
            raise ValueError('Either the time of periapsis or the time of conjunction has to be provided.')

        if t_peri is not None:
            # Give priority to periastron if both are provided
            t0_jd = t_peri.jd
        else:
            # Approximation: superior conjunction f = 90 deg (note: for e>0 and ω, this is only approximate)
            f = np.pi / 2
            e = planet['eccentricity']
            E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2))
            M = E - e * np.sin(E)
            t0_jd = t_conj.jd - (planet['_period'] / (2 * np.pi)) * M

        planet['_t0'] = self.convert_to_bjd(t0_jd) if self.use_bjd else t0_jd
        # --- End of t0 handling ---

        return planet


    def rv_eq(self, f, planet):
        """
        Radial velocity (m/s) of the primary object induced by a planet.

        Parameters
        ----------
        f : np.ndarray or float
            True anomaly (radians).
        planet : dict
            Planet dictionary with the precomputed fields generated in _init_planet.

        Returns
        -------
        v : np.ndarray or float
            RV in m/s (same shape as `f`).
        """
        # Conversion constant AU/day -> m/s
        AU_PER_DAY_TO_MS = (u.au / u.d).to(u.m / u.s)

        e   = float(planet['eccentricity'])
        mt  = self.mass_main.to(u.solMass).value + planet['_m']  # M_total en Msun

        # sqrt( G / (M_tot * a * (1 - e^2)) )  ->  units: AU/day times Msun^-1/2 times AU^-1/2
        k1 = np.sqrt(planet['_grav'] / (mt * planet['_a'] * (1.0 - e * e)))
        # m_p * sin i (in Msun, dimensionless through the sine factor)
        k2 = planet['_m'] * np.sin(planet['_i'])
        # angular factor
        k3 = np.cos(planet['_omega'] + f) + e * np.cos(planet['_omega'])

        v_au_per_day = k1 * k2 * k3
        v_ms = v_au_per_day * AU_PER_DAY_TO_MS

        return v_ms + planet['_gamma']  # gamma is already in m/s


    # def kep_eq(self, e_ano, m_ano, planet):
    #     return e_ano - planet['eccentricity'] * np.sin(e_ano) - m_ano

    def true_anomaly(self, planet, t_jd, tol=1e-12, maxiter=50):
        """
        Return the true anomaly f(t) in radians for a given planet and
        an array of times t_jd (JD or BJD, consistent with self.use_bjd).
        """
        # Ensure a float ndarray
        t = np.asarray(t_jd, dtype=float)

        e = float(planet['eccentricity'])
        P = float(planet['_period'])   # days
        t0 = float(planet['_t0'])      # JD

        # Mean anomaly M(t) and normalization to (-pi, pi] for stability
        M = 2.0 * np.pi * (t - t0) / P
        M = (M + np.pi) % (2.0 * np.pi) - np.pi

        # Circular case: E = M, f = M
        if e == 0.0:
            return M

        # Vectorized Newton-Raphson for E - e sin E - M = 0
        E = M.copy()
        for _ in range(maxiter):
            f = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)
            dE = -f / fp
            E += dE
            if np.max(np.abs(dE)) < tol:
                break

        # Stable conversion to true anomaly
        # cos f = (cosE - e) / (1 - e cosE)
        # sin f = (sqrt(1 - e^2) sinE) / (1 - e cosE)
        cosE = np.cos(E)
        sinE = np.sin(E)
        denom = 1.0 - e * cosE
        cosf = (cosE - e) / denom
        sinf = (np.sqrt(1.0 - e * e) * sinE) / denom
        f_true = np.arctan2(sinf, cosf)  # in (-pi, pi]

        return f_true


    def get_rvs(self, ts, noise=0 * u.m / u.s, include_per_planet=False, rng=None):
        """
        Compute the total radial velocity of the system and return per-planet phases,
        including phase-sorted views to simplify fitting and visualization.

        Parameters
        ----------
        ts : astropy.time.Time | array-like
            Observation dates:
            - scalar or vector astropy.time.Time
            - array/list of Time objects (their .jd values are extracted)
            - array/list of JD values (float)
        noise : astropy.units.Quantity, optional (default 0 m/s)
            Gaussian scatter (1σ) injected into the total RVs. If 0, no noise is added.
        include_per_planet : bool, optional
            If True, also return the RV contribution of each planet.

        Returns
        -------
        jd : np.ndarray
            Dates in JD (float), in chronological order.
        total_rv : np.ndarray
            Sum of all contributions (m/s), with noise if noise>0, in chronological order.
        phases : dict[str, np.ndarray]
            Phase of each planet in [0,1), unsorted (keys 'p0','p1',...).
        phased : dict[str, dict[str, np.ndarray]]
            Phase-sorted views for each planet. For each 'p{i}':
                phased['p{i}']['phase'] -> phase sorted in [0,1)
                phased['p{i}']['rv']    -> total_rv reordered with that index
                phased['p{i}']['jd']    -> jd reordered with that index
                (if include_per_planet=True)
                phased['p{i}']['rv_planet'] -> RV of planet i reordered
        rv_by_planet : dict[str, np.ndarray], optional
            Only if include_per_planet=True. RV of each planet in time order.
        """
        # --- Normalize ts to a JD array (float) ---
        if isinstance(ts, t.Time):
            jd = np.atleast_1d(ts.jd).astype(float)
        elif isinstance(ts, (list, tuple, np.ndarray)):
            if len(ts) > 0 and isinstance(ts[0], t.Time):
                jd = np.array([tk.jd for tk in ts], dtype=float)
            else:
                jd = np.array(ts, dtype=float)
        else:
            raise TypeError("`ts` must be astropy.time.Time or an array/list of Time objects or JD values (floats).")

        # Align with the system convention (BJD or JD)
        if self.use_bjd:
            tref = self.convert_to_bjd(jd)  # JD (converted to BJD if needed)
        else:
            tref = jd

        # --- Local noise level (without touching internal state) ---
        noise_val = noise.to(u.m / u.s).value if (hasattr(noise, "to") or isinstance(noise, u.Quantity)) else float(noise)
        if noise_val < 0:
            raise ValueError("The 'noise' parameter must be >= 0.")

        # --- Accumulators ---
        total_rv = np.zeros_like(jd, dtype=float)
        phases = {}
        phased = {}
        rv_by_planet = {} if include_per_planet else None

        # --- Sum planet contributions and compute phases ---
        for i, planet in enumerate(self.planets):
            # (1) Vectorized true anomaly f(t) for these times
            f_true = self.true_anomaly(planet, tref)

            # (2) RV of planet i using f_true (m/s)
            rv_i = self.rv_eq(f_true, planet)
            total_rv += rv_i

            # (3) Phase of planet i (unsorted), in [0,1)
            P  = planet['_period']  # days
            t0 = planet['_t0']      # JD (consistent with tref)
            phase_i = np.mod((tref - t0) / P, 1.0)

            # (4) Store results
            key = f"p{i}"
            phases[key] = phase_i
            if include_per_planet:
                rv_by_planet[key] = rv_i

        # --- Add Gaussian noise to the total RV if requested ---
        if noise_val > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            total_rv = total_rv + noise_val * rng.normal(size=total_rv.size)

        # --- Build phase-sorted views for EACH planet ---
        for key, phase_i in phases.items():
            order = np.argsort(phase_i)
            phased[key] = {
                'phase': phase_i[order],
                'rv':    total_rv[order],
                'jd':    jd[order],
            }
            if include_per_planet:
                phased[key]['rv_planet'] = rv_by_planet[key][order]

        # --- Output ---
        if include_per_planet:
            return jd, total_rv, phases, phased, rv_by_planet
        else:
            return jd, total_rv, phases, phased


    def obs_dates(self, n_obs, span_days, date_one, rng=None, spam_days=None):
        """Generate a list of observation times.

        Parameters
        ----------
        n_obs : int
            Number of observation epochs.
        span_days : float
            Total time span (days) over which to draw observation times.
        date_one : astropy.time.Time
            Reference start time (UTC).
        rng : numpy.random.Generator, optional
            Random generator for reproducibility.
        spam_days : float, optional
            Deprecated alias for span_days (kept for backwards compatibility).
        """
        if span_days is None and spam_days is not None:
            span_days = spam_days
        if span_days is None:
            raise ValueError("span_days must be provided (or spam_days for backwards compatibility).")

        dt_start = self.convert_to_bjd(date_one.jd) if self.use_bjd else date_one.jd
        if rng is None:
            rng = np.random.default_rng()

        random_dates = rng.uniform(dt_start, dt_start + float(span_days), int(n_obs))
        sorted_dates = np.sort(random_dates)
        time_objects = [t.Time(jd, format="jd", scale="utc") for jd in sorted_dates]
        return time_objects


    def detection_growth_curve_strict(
        self,
        planet_index,
        N,
        span_days,
        start_time,
        sigma_eff,
        *,
        spam_days=None,  # deprecated alias for span_days
        n_iter=200,
        n_min=6,
        tol_rel=0.10,
        fap_method="baluev",     # "baluev" | "bootstrap"
        n_bootstrap=1000,
        freq_pad=4.0,
        min_freq=None,
        max_freq=None,
        rng_seed=None,
        return_all=False
    ):
        """
        En cada prefijo (n=n_min..N):
          1) GLS in [min_freq, max_freq] (centered on f_inj unless explicitly set).
          2) Global peak.
          3) If the peak period falls within [P_inj ± tol_rel*P_inj] -> σ = sigma(FAP).
             Otherwise -> σ = 0.

        Returns a dict with:
          'n', 'sigma_med', 'sigma_mean', 'hit_rate', 'P_inj_days', 'N50', 'N90', ...
          (Optional 'sigma_all', 'hit_all')
        """
        import numpy as np
        import astropy.units as u
        from astropy.timeseries import LombScargle

        rng = np.random.default_rng(rng_seed)

        if span_days is None and spam_days is not None:
            span_days = spam_days

        # Period/frequency of the injected planet
        P_inj = self.planets[planet_index]['orbital_period'].to(u.day).value
        f_inj = 1.0 / P_inj

        # Frequency range
        if (min_freq is None) or (max_freq is None):
            min_freq = max(1e-4, f_inj / freq_pad)
            max_freq = f_inj * freq_pad

        n_vals = np.arange(int(n_min), int(N) + 1, dtype=int)
        sigma_all = np.full((int(n_iter), len(n_vals)), np.nan, float)
        hit_all   = np.zeros((int(n_iter), len(n_vals)), bool)

        for j in range(int(n_iter)):
            dates = self.obs_dates(N, span_days, start_time,rng=rng)
            jd_full, rv_full, _, _ = self.get_rvs(dates, noise=sigma_eff,rng=rng)

            dy_val = (sigma_eff.to(u.m/u.s).value if hasattr(sigma_eff, "to") else float(sigma_eff))
            dy_full = np.full_like(rv_full, dy_val, float)

            for k, n in enumerate(n_vals):
                jd = jd_full[:n]
                rv = rv_full[:n]
                dy = dy_full[:n]

                ls = LombScargle(jd, rv, dy)
                freq, power = ls.autopower(minimum_frequency=min_freq,
                                           maximum_frequency=max_freq,
                                           method='cython')

                kmax = int(np.argmax(power))
                f_best = float(freq[kmax])
                P_best = 1.0 / f_best
                in_window = (abs(P_best - P_inj) <= tol_rel * P_inj)

                if in_window:
                    # p_obs = power[kmax] has already been computed above
                    p_obs = float(power[kmax])

                    if fap_method == "bootstrap":
                        # Use our own bootstrap implementation (without unsupported kwargs)
                        FAP = _bootstrap_fap_maxpower(
                            ls, jd, rv, dy, freq, p_obs,
                            n_bootstrap=int(n_bootstrap),
                            rng=rng
                        )
                    else:
                        # 'baluev' without extra arguments (compatible with your Astropy version)
                        FAP = ls.false_alarm_probability(p_obs, method='baluev')

                    sigma_equiv = safe_sigma_from_fap(FAP, cap_sigma=12.0)
                    hit_all[j, k]   = (sigma_equiv > 0.0)
                    sigma_all[j, k] = sigma_equiv
                else:
                    hit_all[j, k]   = False
                    sigma_all[j, k] = 0.0

        sigma_med  = np.nanmedian(sigma_all, axis=0)
        sigma_mean = np.nanmean(sigma_all, axis=0)
        hit_rate   = hit_all.mean(axis=0)

        N50 = next((int(n) for n, h in zip(n_vals, hit_rate) if h >= 0.50), np.nan)
        N90 = next((int(n) for n, h in zip(n_vals, hit_rate) if h >= 0.90), np.nan)

        out = {
            "n_obs": n_vals,
            "n": n_vals,  # backwards compat
            "sigma_med": sigma_med,
            "sigma_mean": sigma_mean,
            "hit_rate": hit_rate,
            "P_inj_days": P_inj,
            "tol_rel": tol_rel,
            "fap_method": fap_method,
            "n_iter": int(n_iter),
            "span_days": span_days,
            "spam_days": span_days,  # backwards compat
            "freq_range": (min_freq, max_freq),
            "N50": N50,
            "N90": N90,
        }
        if return_all:
            out["sigma_all"] = sigma_all
            out["hit_all"]   = hit_all
        return out

    def detectability_tracker(
        self,
        planet_index,
        n_obs_list,
        span_days,
        start_time,
        sigma_eff,
        *,
        spam_days=None,  # deprecated alias for span_days
        n_iter=200,
        tol_rel=0.10,
        sigma_target=5.0,
        alpha_target=None,       # if None -> norm.sf(sigma_target)
        fap_method="baluev",     # "baluev" | "bootstrap"
        n_bootstrap=1000,
        freq_pad=4.0,
        min_freq=None,
        max_freq=None,
        rng_seed=None
    ):
        """
        Para cada N:
          - Generate n_iter campaigns.
          - GLS in [min_freq, max_freq].
          - The global peak must fall within ±tol_rel around P_inj (hit).
          - If hit, evaluate the FAP and check FAP < alpha_target (equivalent to ≥ sigma_target).
        Returns p_hit, p_det (≥Xσ), and N_min_90 (if it exists).
        """
        import numpy as np
        import astropy.units as u
        from astropy.timeseries import LombScargle
        from scipy.stats import norm

        rng = np.random.default_rng(rng_seed)

        if span_days is None and spam_days is not None:
            span_days = spam_days

        P_inj = self.planets[planet_index]['orbital_period'].to(u.day).value
        f_inj = 1.0 / P_inj

        if alpha_target is None:
            alpha_target = float(norm.sf(sigma_target))

        if (min_freq is None) or (max_freq is None):
            min_freq = max(1e-4, f_inj / freq_pad)
            max_freq = f_inj * freq_pad

        N_arr = np.array(n_obs_list, int)
        p_hit_list, p_det_list = [], []
        fap_med_list, sigma_med_list = [], []

        for N in N_arr:
            hits, dets, faps, sigs = [], [], [], []

            for _ in range(int(n_iter)):
                dates = self.obs_dates(N, span_days, start_time, rng=rng)
                jd, rv, _, _ = self.get_rvs(dates, noise=sigma_eff, rng=rng)

                dy = np.full_like(rv, (sigma_eff.to(u.m/u.s).value if hasattr(sigma_eff, "to") else float(sigma_eff)), float)
                ls = LombScargle(jd, rv, dy)
                freq, power = ls.autopower(minimum_frequency=min_freq,
                                           maximum_frequency=max_freq,
                                           method='cython')

                kmax = int(np.argmax(power))
                f_best = float(freq[kmax])
                P_best = 1.0 / f_best

                hit = (abs(P_best - P_inj) <= tol_rel * P_inj)
                hits.append(hit)

                if hit:
                    p_obs = float(power[kmax])
                    if fap_method == "bootstrap":
                        FAP = _bootstrap_fap_maxpower(
                            ls, jd, rv, dy, freq, p_obs,
                            n_bootstrap=int(n_bootstrap),
                            rng=rng
                        )
                    else:
                        FAP = ls.false_alarm_probability(p_obs, method='baluev')

                    sigma_equiv = safe_sigma_from_fap(FAP, cap_sigma=12.0)
                    det = (FAP < alpha_target)
                    dets.append(det)
                    faps.append(float(FAP))
                    sigs.append(sigma_equiv)
                else:
                    dets.append(False)

            p_hit = np.mean(hits) if hits else 0.0
            p_det = np.mean(dets) if dets else 0.0
            p_hit_list.append(p_hit)
            p_det_list.append(p_det)
            fap_med_list.append(np.median(faps) if len(faps)>0 else np.nan)
            sigma_med_list.append(np.median(sigs) if len(sigs)>0 else np.nan)

        N_min_90 = np.nan
        for N, p in zip(N_arr, p_det_list):
            if p >= 0.90:
                N_min_90 = int(N)
                break

        return {
            "n_obs": N_arr,
            "N": N_arr,  # backwards compat
            "n": N_arr,  # backwards compat
            "p_hit": np.array(p_hit_list, float),
            "p_det": np.array(p_det_list, float),
            "fap_med": np.array(fap_med_list, float),
            "sigma_med": np.array(sigma_med_list, float),
            "P_inj_days": P_inj,
            "tol_rel": tol_rel,
            "sigma_target": sigma_target,
            "alpha_target": alpha_target,
            "fap_method": fap_method,
            "n_iter": int(n_iter),
            "span_days": span_days,
            "spam_days": span_days,  # backwards compat
            "N_min_90": N_min_90,
            "freq_range": (min_freq, max_freq),
        }





    def rv_model(self, dates, mass, eccentricity, planet_index=0):
        """
        Generate the model curve (in phase) for the selected planet by fitting mass
        and eccentricity. Returns (phase_model, rv_model_component), both SORTED
        by the phase of the selected planet.

        Parameters
        ----------
        dates : array-like
            Dates (JD/Time) where the model is evaluated (e.g., a dense grid).
        mass : float
            Planet mass in Earth masses (M_earth).
        eccentricity : float
            Orbital eccentricity of the planet.
        planet_index : int, optional
            Index of the planet to model (default 0).

        Returns
        -------
        phase_model : np.ndarray
            Phases of the selected planet in [0,1), SORTED.
        rv_model_component : np.ndarray
            RV contribution (m/s) of the selected planet, SORTED by that phase.
        """
        if not (0 <= planet_index < len(self.planets)):
            raise IndexError(f"planet_index={planet_index} is out of range (n_planets={len(self.planets)}).")

        # Take the baseline parameters of the selected planet and replace mass/e
        base = self.planets[planet_index]
        planet = {
            'mass': mass * u.earthMass,
            'time_periastron': base.get('time_periastron', None),
            'time_conjunction': base.get('time_conjunction', None),
            'inclination': base['inclination'],
            'argument_periapse': base['argument_periapse'],
            'eccentricity': float(eccentricity),
            'orbital_period': base['orbital_period'],
            'reference_velocity': base.get('reference_velocity', 0 * u.m/u.s),
            'mean_longitude': base.get('mean_longitude', 0 * u.deg),
        }

        # Build a "model" system with ONLY that planet
        model_system = MultiPlanetSystem(
            self.mass_main, [planet],
            use_bjd=getattr(self, 'use_bjd', False),
            location=getattr(self, 'location', None),
            target=getattr(self, 'target', None),
        )

        # Model curve at those dates, noise-free, and request per-planet contribution
        jd_m, rv_m, phases_m, phased_m, rv_by_planet_m = model_system.get_rvs(
            dates, noise=0 * u.m/u.s, include_per_planet=True
        )

        # The planet in this model system is 'p0'
        phase_model = phased_m['p0']['phase']            # sorted
        rv_model_component = phased_m['p0']['rv_planet'] # sorted

        return phase_model, rv_model_component


    def model_for_fit(self, phase_obs, mass, eccentricity, observation_dates_dense, planet_index=0, gamma=0.0):
        """
        Return the model RV (m/s) evaluated at the observed phases 'phase_obs'
        for the selected planet by interpolating the model curve computed on
        'observation_dates_dense'. The interpolation is PERIODIC in [0,1).
        """
        phase_model, rv_model = self.rv_model(
            observation_dates_dense, mass, eccentricity, planet_index=planet_index
        )

        # Periodic interpolation: extend phase and signal by ±1 cycle
        phase_ext = np.concatenate([phase_model - 1.0, phase_model, phase_model + 1.0])
        rv_ext    = np.concatenate([rv_model,         rv_model,    rv_model])

        # np.interp requires 'phase_ext' to be increasing (it already is)
        rv_interp = np.interp(phase_obs, phase_ext, rv_ext)
        return rv_interp + float(gamma)





        

    def precision_tracker(self,
            planet_index,
            n_obs_list,
            span_days,
            start_time,
            sigma_int=None,
            jitter=None,
            beta=1.0,
            fit_mode="mass_gamma",
            bounds=None,
            initial_guess=None,
            n_trials=100,
            n_dense=10000,
            rng_seed=None,
            sigma_eff_known=None,
            spam_days=None,  # deprecated alias for span_days
            verbose=False,
            n_jobs=1,
            chunksize=1,
            mp_start_method='fork',
        ):
        """
        Estimate the precision of the recovered mass for different numbers of observations N.

        It also allows a simplified mode in which the effective measurement error
        (sigma_eff_known) is specified directly. This is useful when typical values
        from analogous stars are available in the literature.

        Parameters
        ----------
        system : MultiPlanetSystem
        planet_index : int
            Index of the planet to evaluate.
        n_obs_list : list[int]
            List of numbers of observations (e.g. [15, 25, 39, 60]).
        span_days : float
            Time window (days) used to generate observations around 'start_time'.
        start_time : astropy.time.Time
            Initial epoch of the campaign (time anchor).
        sigma_int : Quantity [m/s], optional
            Median instrumental internal precision.
        jitter : Quantity [m/s], optional
            Stellar/instrumental jitter to be added in quadrature (default 0).
        beta : float, optional
            Factor accounting for red noise / sampling (default 1.0).
        sigma_eff_known : Quantity [m/s], optional
            Known effective error (from analogous stars). If provided, it is used
            directly and sigma_int, jitter, and beta are ignored.
        fit_mode, bounds, initial_guess, n_trials, n_dense, rng_seed : optional
            Fitting and simulation parameters (as above).
        verbose : bool, optional
            If True, show debug messages when an individual fit fails.
        n_jobs : int or None, optional
            Number of worker processes. If 1, execute serially. If None, use all
            available CPUs.
        chunksize : int, optional
            Chunk size passed to ``ProcessPoolExecutor.map`` when using parallel
            execution.
        mp_start_method : {'fork', 'spawn', 'forkserver'}, optional
            Multiprocessing start method used when ``n_jobs != 1``.

        Returns
        -------
        results : dict
            {
              'N': np.array,
              'mass_precision_pct_med': np.array,
              'mass_precision_pct_p16': np.array,
              'mass_precision_pct_p84': np.array,
              'mass_med': np.array,
              'mass_p16': np.array,
              'mass_p84': np.array
            }
        """
        rng = np.random.default_rng(rng_seed)

        if span_days is None and spam_days is not None:
            span_days = spam_days

        # Effective σ (m/s)
        if sigma_eff_known is not None:
            sigma_eff = sigma_eff_known.to(u.m/u.s)
        else:
            sigma_tot = np.sqrt(sigma_int**2 + jitter**2)
            sigma_eff = (beta * sigma_tot).to(u.m/u.s)

        sigma_eff_mps = float(sigma_eff.to_value(u.m / u.s))

        # Baseline planet values (for a reasonable p0)
        P_days = self.planets[planet_index]['orbital_period'].to(u.day).value

        # Default bounds and p0
        if fit_mode == "mass_gamma":
            if bounds is None:
                bounds = ([0.1, -100.0], [30.0, 100.0])   # mass [M_earth], gamma [m/s]
            if initial_guess is None:
                initial_guess = [self.planets[planet_index]['mass'].to(u.earthMass).value, 0.0]
        elif fit_mode == "mass_e":
            if bounds is None:
                bounds = ([0.1, 0.0], [30.0, 0.3])        # mass [M_earth], e
            if initial_guess is None:
                initial_guess = [self.planets[planet_index]['mass'].to(u.earthMass).value, 0.01]
        elif fit_mode == "mass_only":
            if bounds is None:
                bounds = ([0.1], [30.0])                  # mass [M_earth]
            if initial_guess is None:
                initial_guess = [self.planets[planet_index]['mass'].to(u.earthMass).value]
        else:
            raise ValueError("fit_mode must be 'mass_gamma', 'mass_e', or 'mass_only'.")

        N_arr = np.array(n_obs_list, dtype=int)
        if N_arr.ndim != 1 or N_arr.size == 0:
            raise ValueError("n_obs_list must be a non-empty 1D sequence of integers.")

        if n_jobs is None:
            n_jobs = os.cpu_count() or 1
        n_jobs = int(n_jobs)
        if n_jobs <= 0:
            n_jobs = os.cpu_count() or 1

        tasks = []
        for i_n, N in enumerate(N_arr):
            trial_seeds = rng.integers(0, 2**32 - 1, size=int(n_trials), dtype=np.uint64)
            for seed in trial_seeds:
                tasks.append({
                    'i_n': int(i_n),
                    'N': int(N),
                    'seed': int(seed),
                })

        common_kwargs = {
            'planet_index': int(planet_index),
            'span_days': float(span_days),
            'start_time_jd': float(start_time.jd),
            'sigma_eff_mps': sigma_eff_mps,
            'fit_mode': fit_mode,
            'bounds': tuple(np.asarray(b, dtype=float).tolist() for b in bounds),
            'initial_guess': np.asarray(initial_guess, dtype=float).tolist(),
            'n_dense': int(n_dense),
        }

        if verbose:
            print(
                f"[precision_tracker] tasks={len(tasks)}, n_jobs={n_jobs}, "
                f"chunksize={int(chunksize)}, fit_mode={fit_mode}"
            )

        if n_jobs == 1:
            _init_precision_tracker_worker(self, common_kwargs)
            trial_results = [_precision_tracker_trial_worker(task) for task in tasks]
        else:
            try:
                ctx = mp.get_context(mp_start_method)
            except ValueError:
                ctx = mp.get_context()

            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=ctx,
                initializer=_init_precision_tracker_worker,
                initargs=(self, common_kwargs),
            ) as ex:
                trial_results = list(ex.map(_precision_tracker_trial_worker, tasks, chunksize=int(chunksize)))

        precisions_by_n = [[] for _ in range(len(N_arr))]
        masses_by_n = [[] for _ in range(len(N_arr))]
        success_counts = np.zeros(len(N_arr), dtype=int)

        for res_trial in trial_results:
            i_n = int(res_trial['i_n'])
            if res_trial['ok']:
                precisions_by_n[i_n].append(float(res_trial['precision']))
                masses_by_n[i_n].append(float(res_trial['mass']))
                success_counts[i_n] += 1
            elif verbose:
                print(f"[DEBUG] N={N_arr[i_n]}: curve_fit failed -> {res_trial['error']}")

        prec_med, prec_p16, prec_p84 = [], [], []
        mass_med, mass_p16, mass_p84 = [], [], []

        for precisions, masses in zip(precisions_by_n, masses_by_n):
            if len(precisions) == 0:
                prec_med.append(np.nan)
                prec_p16.append(np.nan)
                prec_p84.append(np.nan)
                mass_med.append(np.nan)
                mass_p16.append(np.nan)
                mass_p84.append(np.nan)
            else:
                precisions = np.asarray(precisions, dtype=float)
                masses = np.asarray(masses, dtype=float)
                prec_med.append(np.nanmedian(precisions))
                prec_p16.append(np.nanpercentile(precisions, 16))
                prec_p84.append(np.nanpercentile(precisions, 84))
                mass_med.append(np.nanmedian(masses))
                mass_p16.append(np.nanpercentile(masses, 16))
                mass_p84.append(np.nanpercentile(masses, 84))

        results = {
            'n_obs': N_arr,
            'N': N_arr,  # backwards compat
            'n': N_arr,  # backwards compat
            'mass_precision_pct_med': np.array(prec_med),
            'mass_precision_pct_p16': np.array(prec_p16),
            'mass_precision_pct_p84': np.array(prec_p84),
            'mass_med': np.array(mass_med),
            'mass_p16': np.array(mass_p16),
            'mass_p84': np.array(mass_p84),
            'P_days': P_days,
            'sigma_eff_mps': sigma_eff_mps,
            'fit_mode': fit_mode,
            'n_trials': int(n_trials),
            'n_trials_success': success_counts,
            'n_jobs': int(n_jobs),
            'chunksize': int(chunksize),
            'mp_start_method': mp_start_method,
        }
        return results


    def _normalize_time_array(self, ts):
        """
        Convert a time input into a 1D JD array (float), preserving the original order.
        """
        if isinstance(ts, t.Time):
            jd = np.atleast_1d(ts.jd).astype(float)
        elif isinstance(ts, (list, tuple, np.ndarray)):
            if len(ts) > 0 and isinstance(ts[0], t.Time):
                jd = np.array([tk.jd for tk in ts], dtype=float)
            else:
                jd = np.array(ts, dtype=float)
        else:
            raise TypeError("`ts` must be astropy.time.Time or an array/list of Time objects or JD values (floats).")

        if jd.ndim != 1:
            raise ValueError("`ts` must be a 1D time series.")
        if np.any(~np.isfinite(jd)):
            raise ValueError("`ts` must contain only finite values.")
        return jd


    def _normalize_rv_inputs(self, ts, rv, rv_err):
        """
        Normalize an RV time series for recovery analysis.

        Parameters
        ----------
        ts : astropy.time.Time or array-like
            Times in Time or JD.
        rv : array-like
            Radial velocities in m/s.
        rv_err : float or array-like
            RV uncertainties in m/s. Can be a scalar or a vector.

        Returns
        -------
        jd : np.ndarray
            Times in JD (float), in the same order as the input.
        rv : np.ndarray
            RVs in m/s.
        rv_err : np.ndarray
            1σ errors in m/s.
        """
        jd = self._normalize_time_array(ts)

        rv = np.asarray(rv, dtype=float)
        if rv.ndim != 1:
            raise ValueError("`rv` must be a 1D array.")
        if jd.shape != rv.shape:
            raise ValueError("`ts` and `rv` must have the same length.")

        if np.isscalar(rv_err):
            rv_err = np.full_like(rv, float(rv_err), dtype=float)
        else:
            rv_err = np.asarray(rv_err, dtype=float)
            if rv_err.shape != rv.shape:
                raise ValueError("`rv_err` must be a scalar or have the same length as `rv`.")

        if np.any(~np.isfinite(jd)) or np.any(~np.isfinite(rv)) or np.any(~np.isfinite(rv_err)):
            raise ValueError("`ts`, `rv`, and `rv_err` must contain only finite values.")
        if np.any(rv_err <= 0):
            raise ValueError("All values of `rv_err` must be > 0.")

        return jd, rv, rv_err


    def rv_semiamplitude_from_msini(self, msini, orbital_period, eccentricity=0.0):
        """
        Compute the expected semiamplitude K for a planet defined by Msin(i).

        Parameters
        ----------
        msini : float or `astropy.Quantity`
            Minimum planetary mass. If a float is provided, it is interpreted in M_earth.
        orbital_period : float or `astropy.Quantity`
            Orbital period. If a float is provided, it is interpreted in days.
        eccentricity : float, optional
            Orbital eccentricity.

        Returns
        -------
        K_mps : float
            Expected semiamplitude in m/s.
        """
        if not np.isfinite(eccentricity) or eccentricity < 0 or eccentricity >= 1:
            raise ValueError("eccentricity must satisfy 0 <= e < 1.")

        if hasattr(msini, 'to'):
            msini_q = msini.to(u.earthMass)
        else:
            msini_q = float(msini) * u.earthMass

        if hasattr(orbital_period, 'to'):
            period_q = orbital_period.to(u.day)
        else:
            period_q = float(orbital_period) * u.day

        mtot = self.mass_main + msini_q
        K = ((2.0 * np.pi * c.G) / period_q) ** (1.0 / 3.0)
        K *= msini_q / (mtot ** (2.0 / 3.0))
        K /= np.sqrt(1.0 - float(eccentricity) ** 2)
        return K.to(u.m / u.s).value


    def inject_planet_in_series(
        self,
        ts,
        msini,
        orbital_period,
        *,
        phase=0.0,
        time_conjunction=None,
        time_periastron=None,
        eccentricity=0.0,
        argument_periapse=90 * u.deg,
        mean_longitude=0 * u.deg,
        reference_velocity=0 * u.m / u.s,
        return_metadata=False,
    ):
        """
        Inject the RV signal of a synthetic planet into a given time series.

        Parameters
        ----------
        ts : astropy.time.Time or array-like
            Observation times (Time or JD).
        msini : float or `astropy.Quantity`
            Minimum planetary mass. If a float is provided, it is interpreted in M_earth.
        orbital_period : float or `astropy.Quantity`
            Orbital period. If a float is provided, it is interpreted in days.
        phase : float, optional
            Orbital phase of conjunction at the minimum time of the series.
            Used only if `time_conjunction` and `time_periastron` are not provided.
        time_conjunction : astropy.time.Time or float, optional
            Time of conjunction. If a float is provided, it is interpreted as JD.
        time_periastron : astropy.time.Time or float, optional
            Time of periapsis. If a float is provided, it is interpreted as JD.
        eccentricity : float, optional
            Orbital eccentricity.
        argument_periapse : `astropy.Quantity`, optional
            Argument of periapsis.
        mean_longitude : `astropy.Quantity`, optional
            Reference mean longitude.
        reference_velocity : `astropy.Quantity`, optional
            Gamma offset of the synthetic planet.
        return_metadata : bool, optional
            If True, also return metadata for the injection.

        Returns
        -------
        rv_inj : np.ndarray
            Injected signal in m/s, evaluated at the input times.

        or, if return_metadata=True

        rv_inj, info : tuple
            info includes `K_inj_mps`, `phase`, `P_days`, and `t0_jd`.
        """
        jd = self._normalize_time_array(ts)

        if hasattr(msini, 'to'):
            mass_q = msini.to(u.earthMass)
        else:
            mass_q = float(msini) * u.earthMass

        if hasattr(orbital_period, 'to'):
            period_q = orbital_period.to(u.day)
        else:
            period_q = float(orbital_period) * u.day

        if not np.isfinite(phase):
            raise ValueError("phase must be finite.")
        phase = float(phase) % 1.0

        if time_conjunction is not None and time_periastron is not None:
            raise ValueError("Use either time_conjunction or time_periastron, not both.")

        if time_conjunction is None and time_periastron is None:
            t0_ref = float(np.min(jd)) + phase * period_q.to_value(u.day)
            time_conjunction = t.Time(t0_ref, format='jd', scale='utc')
        elif time_conjunction is not None and not isinstance(time_conjunction, t.Time):
            time_conjunction = t.Time(float(time_conjunction), format='jd', scale='utc')
        elif time_periastron is not None and not isinstance(time_periastron, t.Time):
            time_periastron = t.Time(float(time_periastron), format='jd', scale='utc')

        planet = {
            'mass': mass_q,
            'inclination': 90 * u.deg,
            'argument_periapse': argument_periapse,
            'eccentricity': float(eccentricity),
            'orbital_period': period_q,
            'mean_longitude': mean_longitude,
            'reference_velocity': reference_velocity,
        }
        if time_periastron is not None:
            planet['time_periastron'] = time_periastron
        else:
            planet['time_conjunction'] = time_conjunction

        tmp_system = MultiPlanetSystem(
            self.mass_main,
            [planet],
            use_bjd=getattr(self, 'use_bjd', False),
            location=getattr(self, 'location', None),
            target=getattr(self, 'target', None),
        )

        _, _, _, _, rv_by_planet = tmp_system.get_rvs(jd, noise=0 * u.m / u.s, include_per_planet=True)
        rv_inj = rv_by_planet['p0']

        if not return_metadata:
            return rv_inj

        info = {
            'K_inj_mps': self.rv_semiamplitude_from_msini(mass_q, period_q, eccentricity=float(eccentricity)),
            'phase': phase,
            'P_days': period_q.to_value(u.day),
            't0_jd': float(tmp_system.planets[0]['_t0']),
        }
        return rv_inj, info


    def _fit_sine_amplitude(self, jd, rv, rv_err, period_days):
        """
        Linear sinusoidal fit y = gamma + a sin(wt) + b cos(wt).

        Used as a fast estimator of the recovered amplitude K for
        approximately circular signals.
        """
        jd = np.asarray(jd, dtype=float)
        rv = np.asarray(rv, dtype=float)
        rv_err = np.asarray(rv_err, dtype=float)
        period_days = float(period_days)

        if period_days <= 0:
            raise ValueError("period_days must be > 0.")

        omega = 2.0 * np.pi / period_days
        tau = jd - np.min(jd)

        X = np.column_stack([
            np.ones_like(tau),
            np.sin(omega * tau),
            np.cos(omega * tau),
        ])
        w = 1.0 / (rv_err ** 2)
        Xw = X * np.sqrt(w)[:, None]
        yw = rv * np.sqrt(w)

        coeffs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        gamma, a_sin, b_cos = coeffs
        k_amp = float(np.hypot(a_sin, b_cos))

        model = gamma + a_sin * np.sin(omega * tau) + b_cos * np.cos(omega * tau)
        return {
            'gamma_mps': float(gamma),
            'K_mps': k_amp,
            'model': model,
            'coeffs': coeffs,
        }


    def recover_periodic_signal(
        self,
        ts,
        rv,
        rv_err,
        *,
        min_period=None,
        max_period=None,
        samples_per_peak=10,
        nyquist_factor=5,
        fap_method='baluev',
        n_bootstrap=1000,
        rng_seed=None,
    ):
        """
        Search for the dominant global peak in an RV time series using GLS.

        Parameters
        ----------
        ts, rv, rv_err : array-like
            Time series and 1σ errors.
        min_period, max_period : float, optional
            Search range in days.
        samples_per_peak : int, optional
            Sampling of the Lomb-Scargle frequency grid.
        nyquist_factor : float, optional
            Nyquist factor passed to `autopower`.
        fap_method : {'baluev', 'bootstrap'}, optional
            Method used to estimate the FAP of the global peak.
        n_bootstrap : int, optional
            Number of bootstrap realizations if `fap_method='bootstrap'`.
        rng_seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        result : dict
            Dictionary containing the recovered period, FAP, equivalent sigma,
            fitted sinusoidal amplitude, and the periodogram grid.
        """
        jd, rv, rv_err = self._normalize_rv_inputs(ts, rv, rv_err)
        rng = np.random.default_rng(rng_seed)

        baseline = float(np.max(jd) - np.min(jd))
        if baseline <= 0:
            raise ValueError("The time series must span a baseline > 0.")

        if min_period is None:
            min_period = max(1e-3, 2.0 * np.median(np.diff(np.sort(jd))) if jd.size > 1 else 0.1)
        if max_period is None:
            max_period = baseline

        min_period = float(min_period)
        max_period = float(max_period)
        if min_period <= 0 or max_period <= 0:
            raise ValueError("min_period and max_period must be > 0.")
        if min_period >= max_period:
            raise ValueError("min_period must be smaller than max_period.")

        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period

        ls = LombScargle(jd, rv, rv_err)
        freq, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=int(samples_per_peak),
            nyquist_factor=float(nyquist_factor),
            method='cython',
        )

        kmax = int(np.argmax(power))
        f_best = float(freq[kmax])
        p_best = 1.0 / f_best
        p_obs = float(power[kmax])

        if fap_method == 'bootstrap':
            FAP = _bootstrap_fap_maxpower(
                ls, jd, rv, rv_err, freq, p_obs,
                n_bootstrap=int(n_bootstrap),
                rng=rng,
            )
        else:
            FAP = ls.false_alarm_probability(p_obs, method='baluev')

        sigma_equiv = safe_sigma_from_fap(FAP, cap_sigma=12.0)
        sine_fit = self._fit_sine_amplitude(jd, rv, rv_err, p_best)

        return {
            'P_best_days': float(p_best),
            'f_best_per_day': f_best,
            'power_best': p_obs,
            'FAP': float(FAP),
            'sigma_equiv': float(sigma_equiv),
            'K_best_mps': float(sine_fit['K_mps']),
            'gamma_best_mps': float(sine_fit['gamma_mps']),
            'period_grid_days': 1.0 / freq,
            'power_grid': power,
            'freq_grid': freq,
        }


    def detectability_map_from_series(
        self,
        ts,
        rv,
        rv_err,
        period_grid,
        msini_grid,
        *,
        n_phase=50,
        phase_mode='grid',
        eccentricity=0.0,
        argument_periapse=90 * u.deg,
        criterion='period+fap',
        period_tol=0.25,
        k_tol=0.25,
        fap_alpha=0.01,
        min_period_search=None,
        max_period_search=None,
        search_pad_factor=1.25,
        samples_per_peak=10,
        nyquist_factor=5,
        fap_method='baluev',
        n_bootstrap=1000,
        rng_seed=None,
        return_trial_details=False,
        verbose=False,
    ):
        """
        Compute a detectability map in the Msin(i) vs. period plane
        for a given RV time series.

        This function does not modify any existing routine in the project. It works
        on a predefined time series (for example, real data, synthetic data, or
        residuals from a previous fit) and injects signals on a grid of periods
        and minimum masses.

        Parameters
        ----------
        ts, rv, rv_err : array-like
            Time series and 1σ errors in m/s.
        period_grid : array-like
            Period grid in days.
        msini_grid : array-like
            Grid of minimum masses. If floats are provided, they are interpreted in M_earth.
        n_phase : int, optional
            Number of injected phases per grid cell.
        phase_mode : {'grid', 'random'}, optional
            Phase sampling mode.
        eccentricity : float, optional
            Fixed eccentricity used in the injections.
        argument_periapse : `astropy.Quantity`, optional
            Argument of periapsis used for the injections.
        criterion : {'period_only', 'period+fap', 'period+K', 'period+fap+K'}, optional
            Recovery criterion.
        period_tol : float, optional
            Relative period tolerance used to declare a recovery.
        k_tol : float, optional
            Relative tolerance on K when using an amplitude-based criterion.
        fap_alpha : float, optional
            FAP threshold when using a significance-based criterion.
        min_period_search, max_period_search : float, optional
            GLS periodogram search range. If not provided, the grid is expanded
            by `search_pad_factor`.
        search_pad_factor : float, optional
            Multiplicative factor used to expand the GLS search range.
        samples_per_peak, nyquist_factor : optional
            Parameters passed to `LombScargle.autopower`.
        fap_method : {'baluev', 'bootstrap'}, optional
            Method used to estimate the FAP.
        n_bootstrap : int, optional
            Number of bootstrap realizations if needed.
        rng_seed : int or None, optional
            Random seed for reproducibility.
        return_trial_details : bool, optional
            If True, store a summary of each individual trial.
        verbose : bool, optional
            If True, print progress by mass row.

        Returns
        -------
        result : dict
            Includes the grid, the recovery_rate map, and per-cell diagnostics.
        """
        jd, rv, rv_err = self._normalize_rv_inputs(ts, rv, rv_err)
        rng = np.random.default_rng(rng_seed)

        period_grid = np.asarray(period_grid, dtype=float)
        if np.any(period_grid <= 0) or period_grid.ndim != 1:
            raise ValueError("`period_grid` must be a 1D array with values > 0.")
        if period_grid.size == 0:
            raise ValueError("`period_grid` cannot be empty.")

        if hasattr(msini_grid, 'to'):
            msini_vals = np.atleast_1d(msini_grid.to_value(u.earthMass)).astype(float)
        else:
            msini_vals = np.asarray(msini_grid, dtype=float)
        if np.any(msini_vals <= 0) or msini_vals.ndim != 1:
            raise ValueError("`msini_grid` must be a 1D array with values > 0.")
        if msini_vals.size == 0:
            raise ValueError("`msini_grid` cannot be empty.")

        phase_mode = phase_mode.strip().lower()
        if phase_mode not in {'grid', 'random'}:
            raise ValueError("phase_mode must be either 'grid' or 'random'.")

        criterion = criterion.strip().lower()
        valid_criteria = {'period_only', 'period+fap', 'period+k', 'period+fap+k'}
        if criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {sorted(valid_criteria)}.")

        if min_period_search is None:
            min_period_search = max(np.min(period_grid) / float(search_pad_factor), 1e-4)
        if max_period_search is None:
            max_period_search = np.max(period_grid) * float(search_pad_factor)

        shape = (msini_vals.size, period_grid.size)
        recovery_rate = np.zeros(shape, dtype=float)
        period_hit_rate = np.zeros(shape, dtype=float)
        median_fap = np.full(shape, np.nan, dtype=float)
        median_sigma = np.full(shape, np.nan, dtype=float)
        median_k_relerr = np.full(shape, np.nan, dtype=float)
        median_p_best = np.full(shape, np.nan, dtype=float)
        trial_details = [] if return_trial_details else None

        for i_m, msini in enumerate(msini_vals):
            if verbose:
                print(f"[detectability_map] mass row {i_m + 1}/{msini_vals.size} (Msin(i)={msini:.4g} M_earth)")

            for j_p, period_days in enumerate(period_grid):
                if phase_mode == 'grid':
                    phases = np.linspace(0.0, 1.0, int(n_phase), endpoint=False)
                else:
                    phases = rng.uniform(0.0, 1.0, int(n_phase))

                recovered_flags = []
                hit_flags = []
                fap_vals = []
                sigma_vals = []
                k_relerr_vals = []
                p_best_vals = []

                for phase in phases:
                    rv_inj, inj_info = self.inject_planet_in_series(
                        jd,
                        msini,
                        period_days,
                        phase=float(phase),
                        eccentricity=float(eccentricity),
                        argument_periapse=argument_periapse,
                        return_metadata=True,
                    )
                    rv_trial = rv + rv_inj

                    rec = self.recover_periodic_signal(
                        jd,
                        rv_trial,
                        rv_err,
                        min_period=float(min_period_search),
                        max_period=float(max_period_search),
                        samples_per_peak=int(samples_per_peak),
                        nyquist_factor=float(nyquist_factor),
                        fap_method=fap_method,
                        n_bootstrap=int(n_bootstrap),
                        rng_seed=int(rng.integers(0, 2**32 - 1)),
                    )

                    rel_period_err = abs(rec['P_best_days'] - period_days) / float(period_days)
                    hit_period = rel_period_err <= float(period_tol)

                    fap_ok = rec['FAP'] < float(fap_alpha)
                    if inj_info['K_inj_mps'] > 0:
                        k_relerr = abs(rec['K_best_mps'] - inj_info['K_inj_mps']) / inj_info['K_inj_mps']
                    else:
                        k_relerr = np.nan
                    k_ok = np.isfinite(k_relerr) and (k_relerr <= float(k_tol))

                    if criterion == 'period_only':
                        recovered = hit_period
                    elif criterion == 'period+fap':
                        recovered = hit_period and fap_ok
                    elif criterion == 'period+k':
                        recovered = hit_period and k_ok
                    else:  # period+fap+k
                        recovered = hit_period and fap_ok and k_ok

                    recovered_flags.append(bool(recovered))
                    hit_flags.append(bool(hit_period))
                    fap_vals.append(float(rec['FAP']))
                    sigma_vals.append(float(rec['sigma_equiv']))
                    k_relerr_vals.append(float(k_relerr) if np.isfinite(k_relerr) else np.nan)
                    p_best_vals.append(float(rec['P_best_days']))

                    if return_trial_details:
                        trial_details.append({
                            'msini_mearth': float(msini),
                            'period_days': float(period_days),
                            'phase': float(phase),
                            'P_best_days': float(rec['P_best_days']),
                            'FAP': float(rec['FAP']),
                            'sigma_equiv': float(rec['sigma_equiv']),
                            'K_inj_mps': float(inj_info['K_inj_mps']),
                            'K_best_mps': float(rec['K_best_mps']),
                            'period_hit': bool(hit_period),
                            'recovered': bool(recovered),
                        })

                recovery_rate[i_m, j_p] = np.mean(recovered_flags) if recovered_flags else np.nan
                period_hit_rate[i_m, j_p] = np.mean(hit_flags) if hit_flags else np.nan
                median_fap[i_m, j_p] = np.nanmedian(fap_vals) if len(fap_vals) else np.nan
                median_sigma[i_m, j_p] = np.nanmedian(sigma_vals) if len(sigma_vals) else np.nan
                median_k_relerr[i_m, j_p] = np.nanmedian(k_relerr_vals) if len(k_relerr_vals) else np.nan
                median_p_best[i_m, j_p] = np.nanmedian(p_best_vals) if len(p_best_vals) else np.nan

        out = {
            'period_grid_days': period_grid,
            'msini_grid_mearth': msini_vals,
            'recovery_rate': recovery_rate,
            'period_hit_rate': period_hit_rate,
            'median_fap': median_fap,
            'median_sigma': median_sigma,
            'median_k_relerr': median_k_relerr,
            'median_p_best_days': median_p_best,
            'criterion': criterion,
            'period_tol': float(period_tol),
            'k_tol': float(k_tol),
            'fap_alpha': float(fap_alpha),
            'n_phase': int(n_phase),
            'phase_mode': phase_mode,
            'eccentricity': float(eccentricity),
            'search_range_days': (float(min_period_search), float(max_period_search)),
            'samples_per_peak': int(samples_per_peak),
            'nyquist_factor': float(nyquist_factor),
            'fap_method': fap_method,
        }
        if return_trial_details:
            out['trial_details'] = trial_details
        return out


    def detectability_map_from_series_parallel(
        self,
        ts,
        rv,
        rv_err,
        period_grid,
        msini_grid,
        *,
        n_phase=50,
        phase_mode='grid',
        eccentricity=0.0,
        argument_periapse=90 * u.deg,
        criterion='period+fap',
        period_tol=0.25,
        k_tol=0.25,
        fap_alpha=0.01,
        min_period_search=None,
        max_period_search=None,
        search_pad_factor=1.25,
        samples_per_peak=10,
        nyquist_factor=5,
        fap_method='baluev',
        n_bootstrap=1000,
        rng_seed=None,
        return_trial_details=False,
        n_jobs=None,
        chunksize=1,
        mp_start_method='fork',
        verbose=False,
    ):
        """
        Parallel version of ``detectability_map_from_series``.

        The scientific logic is intentionally identical to the serial method; the
        only difference is that independent (mass, period) cells are distributed
        across multiple worker processes.

        Parameters
        ----------
        Same as ``detectability_map_from_series`` plus:

        n_jobs : int or None, optional
            Number of worker processes. If None, use all available CPUs.
            If 1, execution is done sequentially through the worker pathway.
        chunksize : int, optional
            Chunk size passed to ``ProcessPoolExecutor.map``.
        mp_start_method : {'fork', 'spawn', 'forkserver'}, optional
            Multiprocessing start method. On Linux, ``'fork'`` is usually fastest.
        verbose : bool, optional
            If True, print a short summary before launching the pool.
        """
        jd, rv, rv_err = self._normalize_rv_inputs(ts, rv, rv_err)
        rng = np.random.default_rng(rng_seed)

        period_grid = np.asarray(period_grid, dtype=float)
        if np.any(period_grid <= 0) or period_grid.ndim != 1:
            raise ValueError("`period_grid` must be a 1D array with values > 0.")
        if period_grid.size == 0:
            raise ValueError("`period_grid` cannot be empty.")

        if hasattr(msini_grid, 'to'):
            msini_vals = np.atleast_1d(msini_grid.to_value(u.earthMass)).astype(float)
        else:
            msini_vals = np.asarray(msini_grid, dtype=float)
        if np.any(msini_vals <= 0) or msini_vals.ndim != 1:
            raise ValueError("`msini_grid` must be a 1D array with values > 0.")
        if msini_vals.size == 0:
            raise ValueError("`msini_grid` cannot be empty.")

        phase_mode = phase_mode.strip().lower()
        if phase_mode not in {'grid', 'random'}:
            raise ValueError("phase_mode must be either 'grid' or 'random'.")

        criterion = criterion.strip().lower()
        valid_criteria = {'period_only', 'period+fap', 'period+k', 'period+fap+k'}
        if criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {sorted(valid_criteria)}.")

        if min_period_search is None:
            min_period_search = max(np.min(period_grid) / float(search_pad_factor), 1e-4)
        if max_period_search is None:
            max_period_search = np.max(period_grid) * float(search_pad_factor)

        if n_jobs is None:
            n_jobs = os.cpu_count() or 1
        n_jobs = int(n_jobs)
        if n_jobs <= 0:
            n_jobs = os.cpu_count() or 1

        shape = (msini_vals.size, period_grid.size)
        recovery_rate = np.zeros(shape, dtype=float)
        period_hit_rate = np.zeros(shape, dtype=float)
        median_fap = np.full(shape, np.nan, dtype=float)
        median_sigma = np.full(shape, np.nan, dtype=float)
        median_k_relerr = np.full(shape, np.nan, dtype=float)
        median_p_best = np.full(shape, np.nan, dtype=float)
        trial_details = [] if return_trial_details else None

        common_kwargs = {
            'eccentricity': float(eccentricity),
            'argument_periapse': argument_periapse,
            'criterion': criterion,
            'period_tol': float(period_tol),
            'k_tol': float(k_tol),
            'fap_alpha': float(fap_alpha),
            'min_period_search': float(min_period_search),
            'max_period_search': float(max_period_search),
            'samples_per_peak': int(samples_per_peak),
            'nyquist_factor': float(nyquist_factor),
            'fap_method': fap_method,
            'n_bootstrap': int(n_bootstrap),
        }

        tasks = []
        for i_m, msini in enumerate(msini_vals):
            for j_p, period_days in enumerate(period_grid):
                if phase_mode == 'grid':
                    phases = np.linspace(0.0, 1.0, int(n_phase), endpoint=False)
                else:
                    phases = rng.uniform(0.0, 1.0, int(n_phase))

                rec_seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=len(phases), dtype=np.uint64)]

                tasks.append({
                    'i_m': int(i_m),
                    'j_p': int(j_p),
                    'msini': float(msini),
                    'period_days': float(period_days),
                    'phases': np.asarray(phases, dtype=float),
                    'rec_seeds': rec_seeds,
                    'return_trial_details': bool(return_trial_details),
                })

        if verbose:
            print(f"[detectability_map_parallel] tasks={len(tasks)}, n_jobs={n_jobs}, chunksize={int(chunksize)}")

        if n_jobs == 1:
            _init_detectability_map_worker(self, jd, rv, rv_err, common_kwargs)
            results = [_detectability_map_cell_worker(task) for task in tasks]
        else:
            try:
                ctx = mp.get_context(mp_start_method)
            except ValueError:
                ctx = mp.get_context()

            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=ctx,
                initializer=_init_detectability_map_worker,
                initargs=(self, jd, rv, rv_err, common_kwargs),
            ) as ex:
                results = list(ex.map(_detectability_map_cell_worker, tasks, chunksize=int(chunksize)))

        results.sort(key=lambda x: (x['i_m'], x['j_p']))

        for res_cell in results:
            i_m = res_cell['i_m']
            j_p = res_cell['j_p']
            recovery_rate[i_m, j_p] = res_cell['recovery_rate']
            period_hit_rate[i_m, j_p] = res_cell['period_hit_rate']
            median_fap[i_m, j_p] = res_cell['median_fap']
            median_sigma[i_m, j_p] = res_cell['median_sigma']
            median_k_relerr[i_m, j_p] = res_cell['median_k_relerr']
            median_p_best[i_m, j_p] = res_cell['median_p_best_days']
            if return_trial_details and res_cell['trial_details']:
                trial_details.extend(res_cell['trial_details'])

        out = {
            'period_grid_days': period_grid,
            'msini_grid_mearth': msini_vals,
            'recovery_rate': recovery_rate,
            'period_hit_rate': period_hit_rate,
            'median_fap': median_fap,
            'median_sigma': median_sigma,
            'median_k_relerr': median_k_relerr,
            'median_p_best_days': median_p_best,
            'criterion': criterion,
            'period_tol': float(period_tol),
            'k_tol': float(k_tol),
            'fap_alpha': float(fap_alpha),
            'n_phase': int(n_phase),
            'phase_mode': phase_mode,
            'eccentricity': float(eccentricity),
            'search_range_days': (float(min_period_search), float(max_period_search)),
            'samples_per_peak': int(samples_per_peak),
            'nyquist_factor': float(nyquist_factor),
            'fap_method': fap_method,
            'n_jobs': int(n_jobs),
            'chunksize': int(chunksize),
            'mp_start_method': mp_start_method,
        }
        if return_trial_details:
            out['trial_details'] = trial_details
        return out


###################################
# === visualization helpers ===
###################################

def n_at_sigma(res, sigma_level=5.0, which="median"):
    """
    Return the first n such that σ(which) >= sigma_level.
    which: "median" | "mean"
    Returns a dict with n_cross (or np.nan if not reached), and the index and σ at that point.
    """
    n = res["n"]
    if which == "median":
        sig = res["sigma_med"]
    elif which == "mean":
        sig = res["sigma_mean"]
    else:
        raise ValueError("which must be 'median' or 'mean'.")

    # First index where the threshold is reached
    idx = np.where(sig >= float(sigma_level))[0]
    if len(idx) == 0:
        return {"n_cross": np.nan, "idx": None, "sigma_at": np.nan}
    i0 = int(idx[0])
    return {"n_cross": int(n[i0]), "idx": i0, "sigma_at": float(sig[i0])}

import matplotlib.pyplot as plt
import numpy as np

def plot_detection_growth_strict(res,
                                 show_goal_sigma=5.0,
                                 mark_sigma_level=None,
                                 mark_which="median",
                                 ax=None,
                                 label=None):
    """
    Plot the evolution of detection significance (sigma) and hit rate
    from detection_growth_curve_strict() results.

    Parameters
    ----------
    res : dict
        Output dictionary from detection_growth_curve_strict().
    show_goal_sigma : float or None
        Draw a horizontal reference line at this sigma level (e.g., 5σ).
    mark_sigma_level : float or None
        Mark the first n where sigma >= mark_sigma_level.
    mark_which : str
        'median' or 'mean' — which curve to use for the significance marker.
    ax : matplotlib.axes.Axes or None
        Optional existing axes.
    label : str or None
        Label for the sigma curves.
    """

    n = res["n"]
    sigma_md = res["sigma_med"]
    sigma_mn = res["sigma_mean"]
    hit      = res["hit_rate"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4))

    # --- Left axis: significance curves ---
    ln1, = ax.plot(n, sigma_md, linewidth=2.2, label=(label or "Median σ (strict)"))
    ln2, = ax.plot(n, sigma_mn, linewidth=1.5, linestyle="--", color="gray", label="Mean σ (strict)")

    # Horizontal target line (e.g., 5σ)
    if show_goal_sigma is not None:
        ax.axhline(show_goal_sigma, linestyle=":", linewidth=1.2, color="red",
                   label=f"{show_goal_sigma:.1f}σ target")

    # --- Optional marker: first n where σ >= chosen level ---
    if mark_sigma_level is not None:
        res_cross = n_at_sigma(res, sigma_level=float(mark_sigma_level), which=mark_which)
        if np.isfinite(res_cross["n_cross"]):
            n_cross = res_cross["n_cross"]
            if mark_which == "median":
                sigma_curve = sigma_md
                color_pt = ln1.get_color()
                label_pt = f"n@{mark_sigma_level:.1f}σ (median) = {n_cross}"
            else:
                sigma_curve = sigma_mn
                color_pt = "gray"
                label_pt = f"n@{mark_sigma_level:.1f}σ (mean) = {n_cross}"

            i_cross = np.where(n == n_cross)[0][0]
            y_cross = sigma_curve[i_cross]

            ax.axvline(n_cross, linestyle="-.", linewidth=1.2, color=color_pt, alpha=0.8)
            ax.plot([n_cross], [y_cross], marker="o", ms=6, color=color_pt)
            ax.text(n_cross + 0.5, y_cross - 0.3, label_pt, color=color_pt, fontsize=9)

    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Detection significance (σ)")
    ax.grid(True, alpha=0.3)

    # --- Right axis: hit rate ---
    ax2 = ax.twinx()
    ax2.plot(n, hit, linewidth=2, color="tab:green", label="Hit rate (global peak at P_inj)")
    ax2.set_ylabel("Hit rate (fraction of realizations)")
    ax2.set_ylim(0, 1.05)

    # N50 and N90 markers
    if not np.isnan(res.get("N50", np.nan)):
        ax2.axvline(res["N50"], linestyle="--", linewidth=1, color="orange", alpha=0.7)
        ax2.text(res["N50"] + 0.5, 0.1, "N50", rotation=90, color="orange", va="bottom")
    if not np.isnan(res.get("N90", np.nan)):
        ax2.axvline(res["N90"], linestyle="--", linewidth=1, color="purple", alpha=0.7)
        ax2.text(res["N90"] + 0.5, 0.1, "N90", rotation=90, color="purple", va="bottom")

    # Combined legend (bottom-right corner)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=9)

    plt.tight_layout()
    return ax



def _grid_edges_from_centers(x, logspace=False):
    """
    Build cell edges from the centers of a 1D grid.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array.")

    if x.size == 1:
        if logspace:
            if x[0] <= 0:
                raise ValueError("logspace=True requires positive values.")
            return np.array([x[0] / np.sqrt(10.0), x[0] * np.sqrt(10.0)], dtype=float)
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)

    if logspace:
        if np.any(x <= 0):
            raise ValueError("logspace=True requires positive values.")
        lx = np.log10(x)
        edges = np.empty(x.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (lx[1:] + lx[:-1])
        edges[0] = lx[0] - 0.5 * (lx[1] - lx[0])
        edges[-1] = lx[-1] + 0.5 * (lx[-1] - lx[-2])
        return 10.0 ** edges

    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[1:] + x[:-1])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges



def plot_detectability_map(
    res_map,
    *,
    key='recovery_rate',
    ax=None,
    cmap='viridis',
    vmin=0.0,
    vmax=1.0,
    log_period=True,
    log_msini=True,
    colorbar=True,
    cbar_label='Detection Probability',
    title=None,
):
    """
    Visualize a map generated by `detectability_map_from_series`.

    Parameters
    ----------
    res_map : dict
        Output of `detectability_map_from_series`.
    key : str, optional
        2D key to display, for example 'recovery_rate'.
    ax : matplotlib.axes.Axes or None
        Existing axes. If None, create a new figure.
    cmap : str, optional
        Matplotlib colormap.
    vmin, vmax : float, optional
        Color scale range.
    log_period, log_msini : bool, optional
        If True, display the axes on a logarithmic scale.
    colorbar : bool, optional
        If True, add a colorbar.
    cbar_label : str or None, optional
        Colorbar label.
    """
    period = np.asarray(res_map['period_grid_days'], dtype=float)
    msini = np.asarray(res_map['msini_grid_mearth'], dtype=float)
    z = np.asarray(res_map[key], dtype=float)

    if z.shape != (msini.size, period.size):
        raise ValueError(
            f"Inconsistent shape for '{key}': expected {(msini.size, period.size)}, got {z.shape}."
        )

    xedges = _grid_edges_from_centers(period, logspace=log_period)
    yedges = _grid_edges_from_centers(msini, logspace=log_msini)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))

    mesh = ax.pcolormesh(xedges, yedges, z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    if log_period:
        ax.set_xscale('log')
    if log_msini:
        ax.set_yscale('log')

    ax.set_xlabel('Orbital period [days]')
    ax.set_ylabel(r'Msin(i) [$M_\oplus$]')
    ax.set_title(title or key)

    if colorbar:
        cb = plt.colorbar(mesh, ax=ax)
        cb.set_label(cbar_label or key)

    plt.tight_layout()
    return ax


#################################
# === persistence helpers ===
#################################

def save_precision_tracker_to_csv(results, filename):
    """
    Save the output of `precision_tracker` to a CSV file.

    Parameters
    ----------
    results : dict
        Dictionary returned by `system.precision_tracker(...)`.
    filename : str or Path
        Output CSV path.
    """

    n = np.asarray(results["n_obs"])

    df = pd.DataFrame({
        "n_obs": n,
        "mass_precision_pct_med": np.asarray(results["mass_precision_pct_med"]),
        "mass_precision_pct_p16": np.asarray(results["mass_precision_pct_p16"]),
        "mass_precision_pct_p84": np.asarray(results["mass_precision_pct_p84"]),
        "mass_med": np.asarray(results["mass_med"]),
        "mass_p16": np.asarray(results["mass_p16"]),
        "mass_p84": np.asarray(results["mass_p84"]),
    })

    # Scalar metadata are repeated in all rows for convenience
    df["P_days"] = results.get("P_days", np.nan)
    df["sigma_eff_mps"] = results.get("sigma_eff_mps", np.nan)
    df["fit_mode"] = results.get("fit_mode", "")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)

    print(f"Results saved to: {filename}")


def load_precision_tracker_from_csv(filename, as_dict=True):
    """
    Load a CSV file previously saved with `save_precision_tracker_to_csv`.

    Parameters
    ----------
    filename : str or Path
        Input CSV path.
    as_dict : bool, default=True
        If True, return a dictionary similar to the original tracker output.
        If False, return the raw pandas DataFrame.

    Returns
    -------
    dict or pandas.DataFrame
    """
    df = pd.read_csv(filename)

    if not as_dict:
        return df

    results = {
        "n_obs": df["n_obs"].to_numpy(dtype=int),
        "N": df["n_obs"].to_numpy(dtype=int),
        "n": df["n_obs"].to_numpy(dtype=int),
        "mass_precision_pct_med": df["mass_precision_pct_med"].to_numpy(dtype=float),
        "mass_precision_pct_p16": df["mass_precision_pct_p16"].to_numpy(dtype=float),
        "mass_precision_pct_p84": df["mass_precision_pct_p84"].to_numpy(dtype=float),
        "mass_med": df["mass_med"].to_numpy(dtype=float),
        "mass_p16": df["mass_p16"].to_numpy(dtype=float),
        "mass_p84": df["mass_p84"].to_numpy(dtype=float),
        "P_days": float(df["P_days"].iloc[0]),
        "sigma_eff_mps": float(df["sigma_eff_mps"].iloc[0]),
        "fit_mode": str(df["fit_mode"].iloc[0]),
    }
    return results
