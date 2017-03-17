import pickle
import numpy as np
import matplotlib.pyplot as plt
from spectral_lines import Measure, Spl, MissingDataError
from sklearn.decomposition import FactorAnalysis
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline as US
from scipy.interpolate import interp1d
from matplotlib.ticker import NullFormatter


# Plotting functions #########################################################


def hist_gauss_fit(data, nbins=10, p0=[1, 0, 1]):
    """Returns a histogram with a Gaussian fit"""
    n, bins, _ = plt.hist(data, bins=nbins)
    binsx = (bins[1:]+bins[:-1])/2
    g = lambda x, *p: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2))
    popt, cov = curve_fit(g, binsx, n, p0=p0, sigma=1/(np.sqrt(n)+0.001))
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, g(x, *popt), 'C3-', linewidth=3, alpha=0.8)
    return popt


SN_LIST = ['1989M',
           '1994S',
           '1995D',
           '2002bo',
           '2002de',
           '2003U',
           '2005de',
           '2005el',
           '2005eq',
           '2005ki',
           '2005lz',
           '2006cj']

# Physics functions ##########################################################


def wave_space(vel):
    """
    Convert velocity into wavelength. Velocity is in km/s.
    """
    return 6355. * np.sqrt((1+vel/3.e5)/(1-vel/3.e5))


def vel_space(wave):
    """
    Returns the feature spectrum in velocity space using the relativistic
    Doppler formula (units are km/s).
    """
    c = 3.e5  # speed of light in km/s
    dl = wave-6355.
    ddl = dl/6355.
    v = c*((ddl+1.)**2.-1.)/((ddl+1.)**2.+1.)
    return v


def vel_from_spec(wave, flux):
    w_abs = wave[np.where(flux == np.min(flux))][0]
    v_abs = vel_space(w_abs)
    return v_abs


##############################################################################


def measure_BSNIP_spline_velocities():
    """
    Measure the velocities with a spline and store results.
    Save this to a initial data file
    """
    print('Measuring velocities with a spline')
    data = {}
    wave, profiles, sne = pickle.load(open('BSNIP_SiII6355_profiles.pkl', 'rb'))
    for profile, sn in zip(profiles, sne):
        spec_data = {}
        var = profile
        try:
            m = Spl([wave, profile, var], sim=True, norm=None)
            w, f, v = m.get_feature_spec()
            spec_data['vspl'] = m.get_line_velocity()[0]
            data[sn] = spec_data
        except MissingDataError:
            print(sn)
            continue
    with open('BSNIP_init_data.pkl', 'wb') as save_file:
        pickle.dump(data, save_file)
    return data


# WFIRST functions ############################################################


def rebin(wave, flux, new_bin_centers):
    """
    Rebin the input spectrum to have the given bin_centers.
    """
    interp = interp1d(wave, flux, bounds_error=False, fill_value=0)
    return new_bin_centers, interp(new_bin_centers)


def add_noise(wave, flux, s2n):
    """
    Add Gaussian noise with the given signal-to-noise characteristic
    """
    sigma = flux/s2n
    noise = np.random.randn(len(flux)) * sigma
    noised_flux = flux + noise
    noised_var = sigma**2
    return wave, noised_flux, noised_var


def wfirst_s2n(wave, flux, z=1):
    wfw, wfsn = np.loadtxt('wfirst_z{:0.2f}.txt'.format(z),
                           unpack=True, usecols=(0, 3))
    rebinned, flux = rebin(wave, flux, wfw)
    rebinned, flux, var = add_noise(rebinned, flux, wfsn)
    return rebinned, flux, var


def gen_WFIRST(z=1, n=10):
    """
    Generate n instances of the spectra in the dataset with WFIRST signal to
    noise and resolution at the specified redshift
    """
    wfirst_data = {}
    print('Generating WFIRST dataset')
    data = pickle.load(open('BSNIP_init_data.pkl', 'rb'))
    for sn, sn_z in np.loadtxt('manifest_clean.txt', dtype=str):
        spec = np.loadtxt(sn, unpack=True)
        wave, flux = spec[:2]
        wave = wave/(1.+float(sn_z))
        wfirst_data[sn] = {'spectra': [], 'vspl': 0}
        try:
            wfirst_data[sn]['vspl'] = data[sn]['vspl']
        except:
            continue
        for i in range(n):
            wfirst_spec = wfirst_s2n(wave, flux, z=z)
            w, f, v = Measure(wfirst_spec, sim=True).get_feature_spec()
            wfirst_data[sn]['spectra'].append([w, f, v])
    fname = 'wfirst_data_z_{}.pkl'.format(z)
    pickle.dump(wfirst_data, open(fname, 'wb'))
    return wfirst_data


def rebin_to_snf(wave, flux):
    """
    Interpolate a spectrum to the SNIFS wavelength bins
    """
    new_bin_centers = np.arange(5001, 8003, 2)
    return rebin(wave, flux, new_bin_centers)


##############################################################################


class BSNIPModel(object):
    def __init__(self, smooth_fac=1):
        # Get profiles from file, if file exists
        try:
            with open('BSNIP_SiII6355_profiles.pkl', 'rb') as profile_file:
                wave, profiles, sne = pickle.load(profile_file)
        except IOError:
            wave, profiles, sne = self.get_profiles()
        self.wave = wave
        self.profiles = profiles
        self.sne = sne
        self.interp_wave = np.arange(min(wave), max(wave), 0.01)

        # Calculate factors that will go into model
        mean, f0, f1, f2 = self.smoothed_factors(self.wave, self.profiles,
                                                 smooth_fac)
        self.mean = mean
        self.f0 = f0
        self.f1 = f1
        self.f2 = f2

        # Dictionary to store data from the analysis
        try:
            self.data = pickle.load(open('BSNIP_init_data.pkl', 'rb'))
        except IOError:
            self.data = measure_BSNIP_spline_velocities()

    def get_profiles(self):
        """
        Get all the line profiles in the training set and save to file.
        """
        profiles = []
        sne = []
        print('Getting profiles from BSNIP training set')
        for sn, z in np.loadtxt('manifest_clean.txt', dtype=str):
            spec = np.loadtxt(sn, unpack=True)
            w, f = spec[:2]
            w = w/(1.+float(z))
            try:
                w, f = rebin_to_snf(w, f)
            except MissingDataError:
                print(sn)
                continue
            v = f
            wave, flux, var = Measure([w, f, v], sim=True).get_feature_spec()
            profiles.append(flux)
            sne.append(sn)
        profiles = np.array(profiles)
        with open('BSNIP_SiII6355_profiles.pkl', 'wb') as save_file:
            pickle.dump([wave, profiles, sne], save_file)
        return wave, profiles, sne

    def smoothed_factors(self, wave, profiles, smooth_fac=1):
        """
        Find the smoothed factors and save to file.
        """
        fa = FactorAnalysis(n_components=3)
        fa.fit(profiles)
        mean_spl = US(wave, fa.mean_, w=1/(smooth_fac*fa.noise_variance_))
        f0_spl = US(wave, fa.components_[0],
                    w=1/(smooth_fac*fa.noise_variance_))
        f1_spl = US(wave, fa.components_[1],
                    w=1/(smooth_fac*fa.noise_variance_))
        f2_spl = US(wave, fa.components_[2],
                    w=1/(smooth_fac*fa.noise_variance_))
        return mean_spl, f0_spl, f1_spl, f2_spl

    def fa_model(self, wave, *params):
        """
        The factor analysis model.
        """
        return self.mean(wave)+params[0]*self.f0(wave)+params[1]*self.f1(wave)+params[2]*self.f2(wave)


class EvalBSNIP(object):

    def __init__(self, model):
        self.model = model
        try:
            fname = 'eval.pkl'
            self.data = pickle.load(open(fname, 'rb'))
        except IOError:
            # Get initial spline velocity measurements from file or calculate
            try:
                fname = 'BSNIP_init_data.pkl'
                self.data = pickle.load(open(fname, 'rb'))
            except IOError:
                self.data = measure_BSNIP_spline_velocities()

            # Fit dataset
            self.fit_dataset()

            # Calculate velocities from the model
            print('Measuring velocities with factors')
            for sn in self.data.values():
                vrec = self.vel_from_reconstruction(sn['load0'], sn['load1'], sn['load2'])
                sn['vmod'] = vrec

            # Save results
            pickle.dump(self.data, open('eval.pkl', 'wb'))

    def fit_model(self, wave, flux, var, return_popt=False):
        """
        Fit the factor analysis model to the input data using least squares.
        """
        p0 = [0, 0, 0]
        popt, cov = curve_fit(self.model.fa_model, wave, flux, p0=p0,
                              sigma=1/np.sqrt(var))
        if return_popt:
            return wave, self.model.fa_model(wave, *popt), popt
        return wave, self.model.fa_model(wave, *popt)

    def fit_dataset(self):
        """
        Fit the model to the data set.
        """
        print('Fitting the FA model to the evalution dataset')
        for sn, profile in zip(self.model.sne, self.model.profiles):
            w, f, v = Spl([self.model.wave, profile, profile],
                          sim=True, norm=None).get_feature_spec()
            fit_w, fit_f, popt = self.fit_model(w, f, v, return_popt=True)
            self.data[sn]['load0'] = popt[0]
            self.data[sn]['load1'] = popt[1]
            self.data[sn]['load2'] = popt[2]

    def vel_from_reconstruction(self, *popt):
        """
        Get a velocity from a reconstruction
        """
        wave = np.arange(min(self.model.wave), max(self.model.wave), 0.01)
        recon_spec = self.model.fa_model(wave, *popt)
        recon_vel = vel_from_spec(wave, recon_spec)
        return recon_vel


class WFIRSTEval(object):

    def __init__(self, model, z=1):
        self.model = model
        self.z = z
        try:
            self.data = pickle.load(open('wfirst_eval.pkl', 'rb'))
        except IOError:
            # Load spectra and spline-measured velocities
            try:
                fname = 'wfirst_data_z_{}.pkl'.format(z)
                self.data = pickle.load(open(fname, 'rb'))
            except IOError:
                self.data = gen_WFIRST(z=z)

            # Fit dataset
            self.fit_dataset()

            # Calculate velocities from the model
            print('Measuring velocities from factors')
            for sn in self.data.values():
                sn['vmod'] = []
                for l0, l1, l2 in zip(sn['load0'], sn['load1'], sn['load2']):
                    vrec = self.vel_from_reconstruction(l0, l1, l2)
                    sn['vmod'].append(vrec)

            # Save results
            pickle.dump(self.data, open('wfirst_eval.pkl', 'wb'))

    def fit_model(self, wave, flux, var, return_popt=False):
        """
        Fit the factor analysis model to the input data using least squares.
        """
        p0 = [0, 0, 0]
        popt, cov = curve_fit(self.model.fa_model, wave, flux, p0=p0,
                              sigma=1/np.sqrt(var))
        if return_popt:
            return wave, self.model.fa_model(wave, *popt), popt
        return wave, self.model.fa_model(wave, *popt)

    def fit_dataset(self):
        """
        Fit the model to the data set.
        """
        print('Fitting the FA model to the evalution dataset')
        for sn in self.data.values():
            sn['load0'] = []
            sn['load1'] = []
            sn['load2'] = []
            for spec in sn['spectra']:
                w, f, v = spec
                _, _, popt = self.fit_model(w, f, v, return_popt=True)
                sn['load0'].append(popt[0])
                sn['load1'].append(popt[1])
                sn['load2'].append(popt[2])

    def vel_from_reconstruction(self, *popt):
        """
        Get a velocity from a reconstruction
        """
        wave = np.arange(min(self.model.wave), max(self.model.wave), 0.01)
        recon_spec = self.model.fa_model(wave, *popt)
        recon_vel = vel_from_spec(wave, recon_spec)
        return recon_vel


def spline_norm_example_plot(sn_name='2005bc'):
    for sn, z in np.loadtxt('manifest_clean.txt', dtype=str):
        if sn_name in sn:
            plt.figure(figsize=(8, 5))
            spec = np.loadtxt(sn, unpack=True)
            wave, flux = spec[:2]
            wave = wave/(1.+float(z))
            plt.subplot(211)
            plt.plot(wave, flux, label='Observed')
            m = Measure([wave, flux, flux], sim=True)
            ws, fs, vs = m.get_snid_norm_spec()
            plt.plot(wave, flux/fs, label='Spline pseudo-cont.')
            plt.ylabel('Observed flux')
            plt.legend()
            plt.subplot(212)
            plt.plot(ws, fs, label='Full spectrum')
            plt.plot(*m.get_feature_spec()[:2], label='Si II feature')
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Normalized flux')
            plt.legend()
            plt.suptitle('SN'+sn_name)
    plt.savefig('example_norm.pdf', bbox_inches='tight')
    plt.close()


def example_reconstruction(model, sn_list=SN_LIST):
    plt.figure(figsize=(12, 12))
    for i in range(12):
        for sn, profile in zip(model.sne, model.profiles):
            plt.subplot(4, 3, i+1)
            if sn_list[i] in sn:
                plt.plot(model.wave, profile, label='Observed')
                popt, cov = curve_fit(model.fa_model, model.wave, profile,
                                      p0=[0, 0, 0])
                plt.plot(model.wave, model.fa_model(model.wave, *popt), 
                         label='Reconstructed')
                plt.text(6300, 0.7, 'SN'+sn_list[i])
                plt.xlabel('Wavelength [$\AA$]')
    plt.subplot(4,3,1); plt.ylabel('Normalized flux')
    plt.subplot(4,3,4); plt.ylabel('Normalized flux')
    plt.subplot(4,3,7); plt.ylabel('Normalized flux')
    plt.subplot(4,3,10); plt.ylabel('Normalized flux')
    plt.savefig('example_reconstruction.pdf', bbox_inches='tight')
    plt.close()


def example_wf_reconstruction(wfeval, sn_list=SN_LIST):
    plt.figure(figsize=(12, 12))
    for i in range(12):
        for sn, profile in zip(wfeval.model.sne, wfeval.model.profiles):
            plt.subplot(4, 3, i+1)
            if sn_list[i] in sn:
                plt.plot(wfeval.model.wave, profile, label='Observed')
                w, f, v = wfeval.data[sn]['spectra'][0]
                plt.errorbar(w, f, yerr=np.sqrt(v), linewidth=0, elinewidth=1.5,
                             marker='.', label='WFIRST')
                popt = [wfeval.data[sn]['load0'][0],
                        wfeval.data[sn]['load1'][0],
                        wfeval.data[sn]['load2'][0]]
                plt.plot(wfeval.model.wave, 
                         wfeval.model.fa_model(wfeval.model.wave, *popt), 
                         label='Reconstructed')
                plt.text(6300, 0.5, 'SN'+sn_list[i])
                plt.xlabel('Wavelength [$\AA$]')
    plt.subplot(4,3,1); plt.ylabel('Normalized flux')
    plt.subplot(4,3,4); plt.ylabel('Normalized flux')
    plt.subplot(4,3,7); plt.ylabel('Normalized flux')
    plt.subplot(4,3,10); plt.ylabel('Normalized flux')
    plt.savefig('example_wfirst_recon.pdf', bbox_inches='tight')
    plt.close()


def model_spl_scatter_hist(data):
    nullfmt = NullFormatter()
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHisty.yaxis.set_major_formatter(nullfmt)

    vspl = np.array([sn['vspl'] for sn in data.values()])
    vmod = np.array([sn['vmod'] for sn in data.values()])
    resids = vspl-vmod

    axScatter.scatter(vspl, resids)
    axScatter.axhline(0, color='k', linestyle='--')
    axScatter.set_ylabel('Velocity residual [km/s]')
    axScatter.set_xlabel('Velocity [km/s]')
    axHisty.hist(resids, orientation='horizontal')
    plt.savefig('training_resids.pdf', bbox_inches='tight')
    plt.close()
    print(np.std(resids))


def wfirst_snf_spl_scatter_hist(data):
    nullfmt = NullFormatter()
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHisty.yaxis.set_major_formatter(nullfmt)

    vspl = np.array([sn['vspl'] for sn in data.values()])
    vmod = np.array([sn['vmod'] for sn in data.values()]).T
    resids = np.array([instance-vspl for instance in vmod]).flatten()

    for i, instance in enumerate(vmod):
        axScatter.scatter(vspl, instance-vspl, c='C0', label='Inst. {}'.format(i))
    axScatter.axhline(0, color='k', linestyle='--')
    axScatter.set_ylabel('Velocity residual [km/s]')
    axScatter.set_xlabel('Velocity [km/s]')
    axHisty.hist(resids, color='C0', orientation='horizontal')
    plt.savefig('wfirst_resids.pdf', bbox_inches='tight')
    plt.close()
    print(np.std(resids))


def correlation_plots(data):
    vspl = [sn['vspl'] for sn in data.values()]
    load0 = [sn['load0'] for sn in data.values()]
    load1 = [sn['load1'] for sn in data.values()]
    load2 = [sn['load2'] for sn in data.values()]
    to_plot = [vspl, load0, load1, load2]
    labels = ['Velocity', 'F1 load coeff.', 'F2 load coeff.', 'F3 load coeff.']
    plt.figure(figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            if j > i:
                continue
            plt.subplot(4, 4, i*4+j+1)
            if i == j:
                plt.hist(to_plot[i])
                if i == 3:
                    plt.xlabel(labels[i])
            else:
                plt.plot(to_plot[j], to_plot[i], '.')
                if i == 3:
                    plt.xlabel(labels[j])
                if j == 0:
                    plt.ylabel(labels[i])
    plt.savefig('correlations.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    spline_norm_example_plot()
    mod = BSNIPModel()
    example_reconstruction(mod)
    e = EvalBSNIP(mod)
    correlation_plots(e.data)
    model_spl_scatter_hist(e.data)
    wf = WFIRSTEval(mod)
    example_wf_reconstruction(wf)
    wfirst_snf_spl_scatter_hist(wf.data)

