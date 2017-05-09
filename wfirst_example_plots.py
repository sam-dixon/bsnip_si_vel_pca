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
    plt.savefig('resolution_z_{:0.2f}.pdf'.format(z), bbox_inches='tight')
    plt.close()


def resolution_plots():
    plt.figure(figsize=(8, 12))
    for i, z in enumerate(np.arange(0.5, 2.0, 0.5)):
        plt.subplot(3, 1, i+1)
        wfw, wfsn = np.loadtxt('wfirst_z{:0.2f}.txt'.format(z),
                               unpack=True, usecols=(0, 3))
        plt.plot(wfw, wfsn, 'o', label='z={}'.format(z))
        plt.xlim(2000, 8000)
        plt.ylim(0, 12)
        plt.legend()
        plt.ylabel('Flux/Flux uncertainty')
    plt.xlabel('Wavelength [$\AA$]')
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
    plt.savefig('example_spectrum_z_{:0.2f}.pdf'.format(z), bbox_inches='tight')


def example_spectra(sn_name='2005bc'):
    plt.figure(figsize=(8, 12))
    for sn, z_obs in np.loadtxt('manifest_clean.txt', dtype=str):
        if sn_name in sn:
            wave, flux = np.loadtxt(sn, unpack=True, usecols=(0, 1))
            wave = wave/(1.+float(z_obs))
            for i, z in enumerate(np.arange(0.5, 2.0, 0.5)):
                plt.subplot(3, 1, i+1)
                plt.plot(wave, flux, 'k-', alpha=0.5, label='Observed')
                w, f, v = wfirst_s2n(wave, flux, z=z)
                plt.errorbar(w, f, yerr=np.sqrt(v), linewidth=0,
                         elinewidth=1.5, marker='.', label='z={}'.format(z))
                plt.xlim(min(wave), max(wave))
                plt.ylabel('Flux')
                plt.legend()
    plt.xlabel('Wavelength [$\AA$]')
    plt.suptitle('SN'+sn_name)
    plt.savefig('example_spectra.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # for z in np.arange(0.5, 1.75, 0.25):
    #     example_spectrum(z=z)
    #     resolution_plot(z=z)
    resolution_plots()
    example_spectra()
