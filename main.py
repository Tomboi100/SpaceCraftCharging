import numpy as np
import pytplot
from matplotlib import pyplot as plt
import pyspedas
from scipy import constants
from scipy.optimize import curve_fit

#task 1
probes='2'
t_range=['2016-12-22/03:42:00','2016-12-22/03:52:00']

pyspedas.mms.edp(trange=t_range,probe=probes,datatype='scpot',data_rate='brst',time_clip=True)
pytplot.tplot(['mms'+probes+'_edp_scpot_brst_l2'])

pyspedas.mms.fpi(trange=t_range,probe=probes,datatype='des-moms',data_rate='brst',time_clip=True)
pytplot.tplot(['mms'+probes+'_des_energyspectr_omni_brst', 'mms'+probes+'_des_bulkv_gse_brst', 'mms'+probes+'_des_numberdensity_brst'])

pyspedas.mms.fgm(trange=t_range,probe=probes,time_clip=True)
pytplot.tplot(['mms'+probes+'_fgm_b_gsm_srvy_l2_btot', 'mms'+probes+'_fgm_b_gsm_srvy_l2_bvec'])

#task 2
pyspedas.tinterpol('mms'+probes+'_edp_scpot_brst_l2','mms'+probes+'_des_numberdensity_brst',method='nearest')
DownSampledSCpot = pyspedas.get_data('mms'+probes+'_edp_scpot_brst_l2-itrp')

elec_Den=pyspedas.get_data('mms'+probes+'_des_numberdensity_brst')
electron_numberdensity=elec_Den.y
electron_numberdensity_times=elec_Den.times
electron_numberdensity_m3=electron_numberdensity*1e6

Te_parallel = pyspedas.get_data('mms' + probes + '_des_temppara_brst')
Te_perpendicular = pyspedas.get_data('mms' + probes + '_des_tempperp_brst')

#units conversion
Te_parallel_K = Te_parallel.y*11600
Te_perpendicular_K = Te_perpendicular.y*11600
Elec_Ttotal_Kelvin = ((2 * Te_perpendicular_K) + Te_parallel_K) / 3

def ElectronThermalCurrent(n,T,V,SpaceCraftArea):
    return SpaceCraftArea*constants.elementary_charge*n*np.sqrt(constants.Boltzmann*T/(2.0*constants.electron_mass*constants.pi))*\
        (1.0+((constants.elementary_charge*V)/(constants.Boltzmann*T)))

ElectronThermalCurrentSeries=np.vectorize(ElectronThermalCurrent)
ElectronThermalCurrentTS=ElectronThermalCurrentSeries(electron_numberdensity_m3, Elec_Ttotal_Kelvin, DownSampledSCpot.y,34.0)
#plotting
plt.figure(0)
plt.plot(DownSampledSCpot.y, ElectronThermalCurrentTS,'o')
plt.xlabel('Spacecraft Potential (V)')
plt.ylabel('Electron Thermal Current (A)')
plt.title('Photocurve (Ie vs Vsc)')
plt.show()

# task 3
def ExpCurve(Vsc, Iph0, Tph0, Iph1, Tph1):
    q = constants.elementary_charge
    kB = constants.Boltzmann
    return Iph0 * np.exp((-1.0 * q * Vsc)/(kB*Tph0)) + Iph1 * np.exp((-1.0*q*Vsc)/(kB*Tph1))

guess = [570.306e-6, 1.5*11604.51812,80.306e-6, 7.*11604.51812]
SCpotArray = DownSampledSCpot.y
popt, pcov = curve_fit(ExpCurve,DownSampledSCpot.y,ElectronThermalCurrentTS, p0=guess, maxfev=50000)
# plotting
plt.figure(0)
plt.plot(DownSampledSCpot.y, ElectronThermalCurrentTS, 'o', label='Data')
Vsc_fitted = np.arange(min(DownSampledSCpot.y), max(DownSampledSCpot.y), 0.05)
Iph_fitted = ExpCurve(Vsc_fitted, popt[0], popt[1], popt[2], popt[3])
plt.plot(Vsc_fitted, Iph_fitted, '-', label='Fitted Curve')
plt.xlabel('Spacecraft Potential (V)')
plt.ylabel('Photoelectron Current (A)')
plt.title('Fitted Photocurve (Iph vs Vsc)')
plt.legend()
plt.show()

#task 4
Iph0, Tph0, Iph1, Tph1 = 0.044, 6744, 0.00033, 35900
SpaceCraftArea = 34.0
SCpot_fullTimes = DownSampledSCpot.times

def electron_density1(T, V, SpaceCraftArea, Iph0, Tph0, Iph1, Tph1):
    return (1.0 / (constants.elementary_charge * SpaceCraftArea)) * \
           np.sqrt(2.0 * constants.electron_mass * constants.pi / (constants.Boltzmann * T)) * \
            (1.0 + ((constants.elementary_charge * V) / (constants.Boltzmann * T)))**-1.0 * \
           (Iph0 * np.exp(-1.0*(constants.elementary_charge * V) / (constants.Boltzmann * Tph0)) +\
            Iph1 * np.exp(-1.0*(constants.elementary_charge * V) / (constants.Boltzmann * Tph1)))

ElectronDensitySeries=np.vectorize(electron_density1)
ElectronDensityTS=ElectronDensitySeries(Elec_Ttotal_Kelvin, DownSampledSCpot.y, SpaceCraftArea, Iph0, Tph0, Iph1, Tph1)

plt.figure(1)
plt.plot(DownSampledSCpot.times-min(SCpot_fullTimes), ElectronDensityTS, '-')
plt.plot(electron_numberdensity_times-min(SCpot_fullTimes), electron_numberdensity_m3, '-')

plt.xlabel('Spacecraft Potential (V)')
plt.ylabel('Electron Density (m^-3)')
plt.title('Electron Density vs. Spacecraft Potential')
plt.show()