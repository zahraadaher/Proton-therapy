"""
Modèle de Physique pour la Thérapie par Protons
==============================================

Ce module contient les calculs physiques pour la thérapie par faisceau de protons,
y compris le calcul du pic de Bragg basé sur l'équation de Bethe-Bloch
pour les protons dans l'eau ou les tissus.
"""

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


class ProtonBeam:
    """
    Modélise un faisceau de protons pour les applications thérapeutiques en utilisant la physique de Bethe-Bloch.
    
    Cette classe calcule le profil de dépôt de dose (courbe de Bragg)
    pour des protons se déplaçant à travers l'eau ou les tissus en utilisant la formule de puissance d'arrêt de Bethe-Bloch.
    """
    
    def __init__(self, medium='water'):
        """
        Initialise le modèle de faisceau de protons avec les constantes physiques.
        
        Paramètres :
        -----------
        medium : str
            Milieu cible ('water' pour équivalent tissu)
        """
        # Constantes physiques
        self.m_e = 0.511  # Masse au repos de l'électron (MeV/c²)
        self.m_p = 938.3  # Masse au repos du proton (MeV/c²)
        self.c = 1.0      # Vitesse de la lumière (normalisée)
        
        # Propriétés du milieu (eau/tissu)
        if medium == 'water':
            self.rho = 1.0         # Densité (g/cm³)
            self.Z_A = 0.5551      # Rapport Z/A pour l'eau
            self.I = 75.0e-6       # Énergie d'excitation moyenne (MeV)
            self.Z = 7.42          # Numéro atomique effectif
            self.A = 18.0          # Nombre de masse effectif

        elif medium == 'air':
            self.rho = 0.001225    # Densité de l'air à 1 atm, 0°C (g/cm³)
            self.Z_A = 0.49919     # Rapport Z/A pour l'air sec
            self.I = 85.7e-6       # Énergie d'excitation moyenne (MeV)
            self.Z = 7.3           # Numéro atomique effectif
            self.A = 14.6          # Nombre de masse effectif
        
        # Rayon classique de l'électron et autres constantes
        self.r_e = 2.818e-13  # cm
        self.N_A = 6.022e23   # Nombre d'Avogadro
        self.Z_over_A = 0.5551  # pour l'eau
    

    def bethe_bloch_stopping_power(self, T):
        """
        Calcule la puissance d'arrêt (dE/dx) en MeV/cm pour une énergie protonique donnée.
        """
        T = np.asarray(T)
        gamma = 1 + T / self.m_p
        beta2 = 1 - 1/(gamma**2)
        beta2 = np.maximum(beta2, 1e-10)

        # Énergie maximale transférable à un électron (MeV)
        T_max = (2 * self.m_e * beta2 * gamma**2) / \
                (1 + 2*gamma*self.m_e/self.m_p + (self.m_e/self.m_p)**2)

        K = 0.307075  # MeV·cm²/g
        ln_term = np.log((2 * self.m_e * beta2 * gamma**2 * T_max) / (self.I**2))
        delta = np.zeros_like(T)  # correction de densité simple ; peut être affinée

        dEdx_mev_per_cm = K * self.Z_over_A  * (self.rho / beta2) * \
                        (0.5*ln_term - beta2 - 0.5*delta)

        # arrêt nucléaire à basse énergie (optionnel, petit)
        nuclear = 0.1 * np.exp(-T/10.0)

        return np.maximum(dEdx_mev_per_cm + nuclear, 0.01)

    def energy_to_range(self, E0):
        """
        Convertit l'énergie initiale en portée dans le milieu (cm).
        """
        num_points = 4000
        E = np.linspace(E0, 0.1, num_points)  # MeV
        dEdx = self.bethe_bloch_stopping_power(E)
        dE = np.diff(E)
        dx = -dE / dEdx[:-1]  # cm
        return np.sum(dx)


    def calculate_bragg_curve(self, E0, depth_cm):
        """
        Calcule la courbe de Bragg pour une énergie initiale donnée et des profondeurs spécifiées.
        """
        depths = np.asarray(depth_cm)
        dose = np.zeros_like(depths, dtype=float)

        # marche du proton jusqu'à l'arrêt
        max_range = self.energy_to_range(E0)
        fine_depths = np.linspace(0, max_range*1.2, 4000)
        current_E = E0

        for i in range(len(fine_depths)-1):
            if current_E <= 0.1:
                break
            d = fine_depths[i+1] - fine_depths[i]
            dEdx = self.bethe_bloch_stopping_power(current_E)
            dE = dEdx * d
            dE = min(dE, max(current_E - 0.1, 0.0))  # ne pas descendre sous le seuil
            mid = 0.5*(fine_depths[i] + fine_depths[i+1])
            j = np.searchsorted(depths, mid)
            if 0 <= j < len(dose):
                dose[j] += dE
            current_E -= dE

        # optionnel : convolution avec Gaussienne pour modéliser la straggling
        if dose.sum() > 0:
            dose /= dose.max()
            sigma = 0.015 * max_range  # ~1,5% de la portée
            if sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                step = np.mean(np.diff(depths))
                s_bins = max(sigma/step, 0.5)
                dose = gaussian_filter1d(dose, s_bins, mode='nearest')
                dose /= dose.max()

        return dose

    
    # def get_peak_position(self, initial_energy_mev):
    #     """
    #     Retourne la profondeur du pic de Bragg pour une énergie initiale donnée.
    #     """
    #     total_range = self.energy_to_range(initial_energy_mev)
    #     peak_position = 0.98 * total_range  # ~98% de la portée
    #     return peak_position
    
    def get_detailed_physics_info(self, energy_mev):
        """
        Fournit des informations physiques détaillées à des fins éducatives.
        """
        gamma = 1 + energy_mev / self.m_p
        beta = np.sqrt(1 - 1/gamma**2)
        velocity = beta * 3e10  # cm/s
        
        stopping_power = self.bethe_bloch_stopping_power(energy_mev)
        range_cm = self.energy_to_range(energy_mev)
        
        return {
            'energy_mev': energy_mev,
            'velocity_cm_per_s': velocity,
            'beta': beta,
            'gamma': gamma,
            'stopping_power_mev_per_cm': stopping_power,
            'range_cm': range_cm,
            'peak_depth_cm': self.get_peak_position(energy_mev)
        }
    
    def get_peak_position(self, initial_energy_mev, depth_resolution=0.01):
        """
        Calculer la position du pic de Bragg en trouvant le maximum de la courbe de Bragg.

        Arguments :
            initial_energy_mev : énergie initiale du proton en MeV
            depth_resolution : pas de profondeur en cm pour l'échantillonnage

        Retour :
            peak_depth : profondeur du pic de Bragg en cm
        """
        # 1. Obtenir la portée totale pour définir le tableau de profondeurs
        total_range = self.energy_to_range(initial_energy_mev)
        depths = np.arange(0, total_range, depth_resolution)

        # 2. Calculer la courbe de Bragg pour ces profondeurs
        dose = self.calculate_bragg_curve(initial_energy_mev, depths)

        # 3. Trouver la profondeur correspondant à la dose maximale
        peak_index = np.argmax(dose)
        peak_depth = depths[peak_index]

        return peak_depth




class Tumor:
    """
    Représente une tumeur cible pour la thérapie par protons.
    """
    
    def __init__(self, center_depth_cm, width_cm=1.0):
        """
        Initialise la tumeur.
        """
        self.center_depth = center_depth_cm
        self.width = width_cm
        
    def get_boundaries(self):
        """
        Retourne les limites de la tumeur.
        """
        half_width = self.width / 2
        return (self.center_depth - half_width, 
                self.center_depth + half_width)
    
    def is_hit_by_peak(self, peak_depth, tolerance=0.5):
        """
        Vérifie si le pic de Bragg atteint la tumeur dans une tolérance donnée.
        """
        return abs(peak_depth - self.center_depth) <= tolerance


def create_depth_array(max_depth_cm=25, num_points=500):
    """
    Crée un tableau de profondeurs pour les calculs.
    """
    return np.linspace(0, max_depth_cm, num_points)
