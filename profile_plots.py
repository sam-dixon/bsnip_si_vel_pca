import pickle
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as US
from matplotlib import cm, rcParams
import numpy as np

rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16

fa = FactorAnalysis(n_components=3)
colors = [cm.coolwarm(x) for x in np.linspace(0, 1, 15)]


filename = 'BSNIP_SiII6355_profiles.pkl'
w, profiles, sne = pickle.load(open(filename, 'rb'))
fa.fit(profiles)
transformed = fa.fit_transform(profiles)


Z = [[0,0],[0,0]]
levels0 = np.linspace(min(transformed[:, 0]), max(transformed[:, 0]), 15)
cb0 = plt.contourf(Z, levels0, cmap=cm.coolwarm)
plt.close()
levels1 = np.linspace(min(transformed[:, 1]), max(transformed[:, 1]), 15)
cb1 = plt.contourf(Z, levels1, cmap=cm.coolwarm)
plt.close()
levels2 = np.linspace(min(transformed[:, 2]), max(transformed[:, 2]), 15)
cb2 = plt.contourf(Z, levels2, cmap=cm.coolwarm)
plt.close()

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.hist(transformed[:, 0], bins=levels0)
plt.title('Factor 1', fontsize=18)
plt.xlim(levels0[0], levels0[-1])
plt.ylim(0, 22)
plt.ylabel('Training set frequency', fontsize=16)
cb0 = plt.colorbar(cb0, orientation='horizontal')
cb0.set_ticks([])
cb0.set_label('Coefficient', fontsize=16)
plt.subplot(232)
plt.hist(transformed[:, 1], bins=levels1)
plt.xlim(levels1[0], levels1[-1])
plt.ylim(0, 22)
plt.title('Factor 2', fontsize=18)
cb1 = plt.colorbar(cb1, orientation='horizontal')
cb1.set_label('Coefficient', fontsize=16)
cb1.set_ticks([])
plt.subplot(233)
plt.hist(transformed[:, 2], bins=levels2)
plt.xlim(levels2[0], levels2[-1])
plt.ylim(0, 22)
plt.title('Factor 3', fontsize=18)
cb1 = plt.colorbar(cb2, orientation='horizontal')
cb1.set_label('Coefficient', fontsize=16)
cb1.set_ticks([])
plt.subplot(234)
for i, c in enumerate(np.linspace(min(transformed[:, 0]), max(transformed[:, 0]), 15)):
    spl = US(w, fa.components_[0], w=1/fa.noise_variance_)
    mean_spl = US(w, fa.mean_, w=1/fa.noise_variance_)
    plt.plot(w, mean_spl(w)+spl(w)*c, color=colors[i])
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.ylabel('Normalized flux', fontsize=16)
plt.subplot(235)
for i, c in enumerate(np.linspace(min(transformed[:, 1]), max(transformed[:, 1]), 15)):
    spl = US(w, fa.components_[1], w=1/fa.noise_variance_)
    mean_spl = US(w, fa.mean_, w=1/fa.noise_variance_)
    plt.plot(w, mean_spl(w)+spl(w)*c, color=colors[i])
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.subplot(236)
for i, c in enumerate(np.linspace(min(transformed[:, 2]), max(transformed[:, 2]), 15)):
    spl = US(w, fa.components_[2], w=1/fa.noise_variance_)
    mean_spl = US(w, fa.mean_, w=1/fa.noise_variance_)
    plt.plot(w, mean_spl(w)+spl(w)*c, color=colors[i])
plt.xlabel('Wavelength [$\AA$]', fontsize=16)
plt.tight_layout()
# plt.savefig('bsnip_factors.pdf', bbox_inches='tight')
plt.show()