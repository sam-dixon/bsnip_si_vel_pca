import numpy as np
import matplotlib.pyplot as plt
from run_analysis import wfirst_s2n


def resolution_plot(z=1):
    plt.figure(figsize=(8, 5))
    wfw, wfsn = np.loadtxt('wfirst_z{:0.2f}.txt'.format(z),
                           unpack=True, usecols=(0, 3))
    plt.plot(wfw, wfsn, 'o')
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux/Flux uncertainty')
    plt.savefig('resolution.pdf', bbox_inches='tight')
    plt.close()


def example_spectrum(sn_name='2005bc', z=1):
    plt.figure(figsize=(8, 5))
    for sn, z_obs in np.loadtxt('manifest_clean.txt', dtype=str):
        if sn_name in sn:
            spec = np.loadtxt(sn, unpack=True)
            wave, flux = spec[:2]
            wave = wave/(1.+float(z_obs))
            plt.plot(wave, flux, 'k-', alpha=0.5, label='Observed')
            w, f, v = wfirst_s2n(wave, flux, z=z)
            plt.errorbar(w, f, yerr=np.sqrt(v), linewidth=0,
                         elinewidth=1.5, marker='.', label='Simulated')
            plt.xlim(min(wave), max(wave))
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('SN'+sn_name)
    plt.savefig('example_spectrum.pdf', bbox_inches='tight')


if __name__ == '__main__':
    resolution_plot()
    example_spectrum()
