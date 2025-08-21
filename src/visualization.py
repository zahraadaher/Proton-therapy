import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from IPython.display import clear_output


class ProtonTherapyPlotter:
    """
    Gère tous les tracés pour l'expérience de thérapie par protons.
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialiser le traceur.
        
        Paramètres :
        -----------
        figsize : tuple
            Taille de la figure (largeur, hauteur) en pouces
        """
        self.figsize = figsize
        self.configurer_style()
        
    def configurer_style(self):
        """Configurer le style de matplotlib pour des tracés clairs."""
        plt.style.use('default')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        
    def plot_bragg_curve_with_tumor(self, depths, dose, tumor, peak_depth, 
                                energy_mev, max_depth=25, physics_info=None):
        """
        Trace la courbe de Bragg avec l'emplacement de la tumeur et les informations physiques.
        
        Paramètres :
        -----------
        depths : array-like
            Profondeurs en cm
        dose : array-like  
            Valeurs de dose (relative)
        tumor : Objet Tumor
            La tumeur à cibler
        peak_depth : float
            Profondeur du pic de Bragg en cm
        energy_mev : float
            Énergie du faisceau en MeV
        max_depth : float
            Profondeur maximale à afficher
        physics_info : dict
            Informations physiques détaillées
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # Tracé principal de la courbe de Bragg
        ax1.plot(depths, dose, 'b-', linewidth=3, label='Profil de dose Bethe-Bloch')
        ax1.fill_between(depths, dose, alpha=0.3, color='blue')
        
        # Marquer le pic de Bragg
        ax1.axvline(peak_depth, color='red', linestyle='--', linewidth=2, 
                label=f'Pic de Bragg ({peak_depth:.1f} cm)')
        
        # Montrer l'emplacement de la tumeur
        tumor_start, tumor_end = tumor.get_boundaries()
        tumor_height = max(dose) * 0.8 if max(dose) > 0 else 0.8
        
        # Rectangle de la tumeur
        tumor_rect = patches.Rectangle((tumor_start, 0), 
                                    tumor.width, tumor_height,
                                    facecolor='red', alpha=0.4, 
                                    edgecolor='darkred', linewidth=2,
                                    label='Tumeur')
        ax1.add_patch(tumor_rect)
        
        # Ligne du centre de la tumeur
        ax1.axvline(tumor.center_depth, color='darkred', linestyle='-', 
                linewidth=3, alpha=0.8)
        
        # Vérifier si le pic atteint la tumeur
        hit = tumor.is_hit_by_peak(peak_depth)
        hit_text = "🎯 CIBLE ATTEINTE !" if hit else "❌ CIBLE MANQUÉE"
        hit_color = 'green' if hit else 'red'
        
        ax1.text(0.02, 0.95, hit_text, transform=ax1.transAxes, 
                fontsize=16, fontweight='bold', color=hit_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Distance par rapport à la cible
        distance = abs(peak_depth - tumor.center_depth)
        ax1.text(0.02, 0.87, f'Distance à la tumeur : {distance:.1f} cm', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Informations physiques
        if physics_info:
            physics_text = f"β = {physics_info['beta']:.3f}\nPortée = {physics_info['range_cm']:.1f} cm"
            ax1.text(0.98, 0.95, physics_text, transform=ax1.transAxes, 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax1.set_xlabel('Profondeur dans le tissu (cm)')
        ax1.set_ylabel('Dose relative (Bethe-Bloch)')
        ax1.set_title(f'Thérapie par protons : Modèle Bethe-Bloch (Énergie : {energy_mev:.0f} MeV)')
        ax1.set_xlim(0, max_depth)
        ax1.set_ylim(0, max(dose) * 1.1 if max(dose) > 0 else 1.1)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Diagramme simple de coupe du patient
        ax2.set_xlim(0, max_depth)
        ax2.set_ylim(-1, 1)
        
        # Dessiner le contour du corps
        body_patch = patches.Rectangle((0, -0.8), max_depth, 1.6, 
                                    facecolor='lightgray', alpha=0.3,
                                    edgecolor='gray')
        ax2.add_patch(body_patch)
        
        # Montrer l'entrée du faisceau
        ax2.arrow(-1, 0, 1.5, 0, head_width=0.2, head_length=0.5, 
                fc='blue', ec='blue', linewidth=3)
        ax2.text(-1, 0.4, 'Faisceau\nde protons', ha='center', fontsize=10, 
                color='blue', fontweight='bold')
        
        # Montrer la tumeur en coupe
        tumor_cross = patches.Rectangle((tumor_start, -0.3), tumor.width, 0.6,
                                        facecolor='red', alpha=0.6,
                                        edgecolor='darkred', linewidth=2)
        ax2.add_patch(tumor_cross)
        ax2.text(tumor.center_depth, 0, 'TUMEUR', ha='center', va='center',
                fontweight='bold', color='white', fontsize=10)
        
        # Montrer la position du pic
        ax2.axvline(peak_depth, color='red', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Profondeur dans le patient (cm)')
        ax2.set_ylabel('')
        ax2.set_title('Coupe du patient (tissu équivalent eau)')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
            
    def plot_stopping_power_curve(self, energies, stopping_powers):
        """
        Tracer la courbe de pouvoir d'arrêt selon Bethe-Bloch.
        
        Paramètres :
        -----------
        energies : array-like
            Énergies des protons en MeV
        stopping_powers : array-like
            Pouvoirs d'arrêt en MeV/cm
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(energies, stopping_powers, 'b-', linewidth=3, 
                label='Pouvoir d\'arrêt Bethe-Bloch')
        plt.xlabel('Énergie du proton (MeV)')
        plt.ylabel('Pouvoir d\'arrêt (MeV/cm)')
        plt.title('Pouvoir d\'arrêt Bethe-Bloch dans l\'eau/tissu')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ajouter une annotation physique
        plt.text(0.7, 0.8, r'$\frac{dE}{dx} \propto \frac{1}{\beta^2} \ln\left(\frac{2m_e c^2 \beta^2 \gamma^2}{I}\right)$')

        
    def plot_energy_vs_range(self, energies, ranges):
        """
        Tracer la relation entre l'énergie des protons et leur portée.
        
        Paramètres :
        -----------
        energies : array-like
            Énergies du faisceau en MeV
        ranges : array-like
            Portées correspondantes en cm
        """
        plt.figure(figsize=(8, 6))
        plt.plot(energies, ranges, 'b-', linewidth=3)
        plt.xlabel('Énergie du proton (MeV)')
        plt.ylabel('Portée dans le tissu (cm)')
        plt.title('Énergie vs Portée des protons dans le tissu')
        plt.grid(True, alpha=0.3)
        plt.show()



def display_instructions():
    """Afficher les instructions de l'expérience pour les étudiants."""
    instructions = """
    🔬 EXPÉRIENCE DE THÉRAPIE PROTONIQUE 🔬

    MISSION : Ajustez l'énergie du faisceau de protons pour atteindre la tumeur avec le pic de Bragg !

    INSTRUCTIONS :
    1. Utilisez le curseur d'énergie ci-dessous pour modifier l'énergie du faisceau de protons
    2. Observez comment la position du pic de Bragg change dans le graphique
    3. Essayez de positionner le pic exactement sur la tumeur rouge
    4. Plus vous vous rapprochez, meilleur est le traitement !

    FAITS PHYSIQUES :
    • Énergie plus élevée → les protons pénètrent plus profondément
    • Énergie plus faible → les protons s'arrêtent plus tôt
    • Le pic de Bragg est l'endroit où les protons déposent le plus d'énergie
    • Les vrais médecins utilisent ceci pour cibler les tumeurs avec précision !

    Prêt à sauver la situation ? 🦸‍♂️🦸‍♀️
    """

    
    print(instructions)


def create_summary_stats(peak_depth, tumor_center, energy_mev):
    """
    Crée un résumé de la configuration actuelle du faisceau.
    
    Paramètres :
    -----------
    peak_depth : float
        Profondeur actuelle du pic de Bragg
    tumor_center : float  
        Centre de la tumeur
    energy_mev : float
        Énergie du faisceau
        
    Retour :
    --------
    str
        Chaîne formatée du résumé
    """
    distance = abs(peak_depth - tumor_center)
    accuracy = max(0, 100 * (1 - distance / 2))  # % de précision dans 2 cm
    
    summary = f"""
    ÉTAT ACTUEL DU FAISCEAU :
    ━━━━━━━━━━━━━━━━━━━━
    Énergie : {energy_mev:.0f} MeV
    Profondeur du pic : {peak_depth:.1f} cm  
    Profondeur de la tumeur : {tumor_center:.1f} cm
    Distance : {distance:.1f} cm
    Précision : {accuracy:.0f}%
    
    {"🎯 EXCELLENT !" if distance < 0.5 else 
     "🟡 BON" if distance < 1.0 else 
     "🔴 AJUSTEMENT NÉCESSAIRE"}
    """

    
    return summary