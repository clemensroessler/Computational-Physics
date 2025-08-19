"""Quantenmechanik von 1D-Potentialen I: Eigenwerte, Eigenfunktionen
Das Programm bestimmt fur ein Teilchen im asymmetrischen 
Doppelmuldenpotential alle Eigenenergien bis zu einer Energie-Grenze und die 
zugehörigen Eigenfunktionen. Es stellt diese dann grafisch zusammen mit dem 
Potential dar. 
"""

"""
Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs
Code, die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die
nächste Zeile kommentiert/entfernt werden.
"""

# %matplotlib ipympl




import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import functools
def doppelmulde(x, A=0.055):
    """Funktion zur Berechnung des Doppelmuldenpotentials an Stelle x für 
    Parameter A."""
    return x ** 4 - x ** 2 + A * x


def eigen_berechnungen(potential, x_min=-3, x_max=3, anz_dim=200, h_eff=0.06,
                       energie_grenze=0.15):
    """
    Funktion zur Berechnung der Eigenenergien und Eigenzustände 
    für das gegebene Potential mittels der Schrödingergleichung.

    Args:
    - potential: Funktion des Potentials
    - x_min: Minimale x-Wert des Bereichs
    - x_max: Maximale x-Wert des Bereichs
    - anz_dim: Anzahl der Dimensionen für die Diskretisierung
    - h_eff: Effektive Planck-Konstante
    - energie_grenze: Grenzwert für die relevanten Eigenenergien

    Returns:
    - Eigenenergien, Eigenzustände, x-Werte
    """
    # Berechnung der Schrittweite
    delta_x = (x_max - x_min) / (anz_dim + 1)
    # Erzeugung des Arrays für die diskreten x-Werte
    args = x_min + (np.arange(anz_dim) + 1) * delta_x
    # Berechnung des Potential-Arrays
    V_array = potential(args)
    # Berechnung der Konstante für die diskrete Schrödingergleichung
    z_konstante = h_eff ** 2 / (2 * delta_x ** 2)
    # Erzeugung des Arrays für die Nebendiagonalen
    z_array = np.ones(anz_dim - 1) * z_konstante
    # Aufbau der Matrix für die diskrete Schrödingergleichung
    eigenmatrix = np.diag(V_array + 2 * z_konstante) + \
        np.diag(-z_array, k=-1) + np.diag(-z_array, k=1)
    # Berechnung der Eigenwerte und Eigenvektoren
    eigen_energien, eigenvektoren = eigh(
        eigenmatrix, subset_by_value=[-np.inf, energie_grenze])
    # Normierung der Eigenvektoren
    normierte_ev = eigenvektoren * delta_x ** (-1 / 2)
    return eigen_energien, normierte_ev, args


def skalieren_und_plotten(eigen_energien, normierte_ev, args, potential):
    """
    Funktion zur Skalierung der Eigenzustände und Erstellung des Plots.

    Args:
    - eigen_energien: Eigenenergien
    - normierte_ev: Normierte Eigenzustände
    - args: x-Werte
    - potential: Funktion des Potentials
    """
    # Berechnung der Differenzen der Eigenenergien
    energie_diff = np.diff(eigen_energien)
    # Berechnung des maximalen Werts der normierten Eigenzustände
    max_energie = np.max(np.abs(normierte_ev))
    # Berechnung des mittleren Differenzwertes der Eigenenergien
    mean_energie_diff = np.mean(energie_diff)

    # Skalierung der Eigenzustände
    skalierte_ev = normierte_ev * mean_energie_diff / (2 * max_energie)
    # Berechnung der Potentialkurve
    potential_kurve = potential(args)

    # Festlegung der Achsengrenzen für x und Energie
    x_achsengrenze = [np.min(args), np.max(args)]
    E_achsengrenze = [np.min(potential_kurve), np.max(
        eigen_energien) + mean_energie_diff / 2]

    # Erstellen des Plots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Festlegung der Achsenbereiche
    ax.set_xlim(*x_achsengrenze)
    ax.set_ylim(*E_achsengrenze)

    # Beschriftung der Achsen und des Titels
    ax.set_xlabel("Stelle x", fontsize=14)
    ax.set_ylabel("Energie E", fontsize=14)
    ax.set_title("Teilchen in Doppelmuldenpotential", fontsize=16)

    # Hinzufügen von Rasterlinien
    ax.grid(True)

    # Plotten der Eigenzustände und Zählen der Extrema
    for i in range(eigen_energien.size):
        ax.plot(args, skalierte_ev[:, i] + eigen_energien[i],
                ls="-", label=f"Eigenzustand {i+1}")
    # Plotten der Potentialkurve
    ax.plot(args, potential_kurve, linestyle="--",
            color="black", label="Potentialkurve")

    # Hinzufügen einer Legende
    ax.legend(fontsize=12, loc="lower right")

    plt.show()


def main():
    """Hauptprogramm. Aufruf für verschiedene Parameter."""
    print(__doc__)

    # Ausgabe der relevanten Parameter
    print("""Verwendete Parameter: Das betrachtete Intervall wurde von x = -2 
          bis x = 2 gewählt. Die Dimension der Matrix zur Berechnung der 
          Schrödingergleichung (SGL) beträgt 200. Das effektives Plancksche 
          Wirkungsquantum h_eff wurde mit 0.06 vorgegeben und der 
          Potentialparameter A wurde ebenfalls gegeben mit A = 0.055. 
          Die Energiegrenze der Eigenenrgien wurde bei 0.15 gesetzt.""")

    # Ausgabe der Benutzerführung
    print("""Sie sehen ein Diagramm welches das Potential in gestrichelter 
          schwarzer Linie darstellt. Die Eigenfunktionen sind auf Höhe der  
          zugöhrigen Eigenenergien eingezeichnet. Die Eigenenergien und das 
          Potential sind akurat gegenüber der Energieachse skaliert. Die 
          Eigenfunktionen hingegen sind alle mit einem Faktor so skaliert das 
          sie gut sichtbar sind. In der Legende sind die Eigenfunktionen 
          aufsteigend in ihrer Energie geordnet, sodass die erste 
          Eigenfunktion den Grundzustand darstellt, also jene Eigenfunktion 
          welche die niedrigste Eigenenergie besitzt.""")

    # Festlegung der Parameter

    # Intervall in welchem die Schrödingergleichung (SGL) gelöst wird
    x_min = -2
    x_max = 2
    # gibt die Dimensionen der Matrix zur Berechnung der SGL an.
    anz_dim = 200
    # effektives Plancksches Wirkungsquantum
    h_eff = 0.06
    # Potentialparameter
    A = 0.055
    # Potentialfunktion mit festem A
    potential_von_A = functools.partial(doppelmulde, A=A)
    # Grenze über welcher Eigenenergien nicht betrachtet werden
    energie_grenze = 0.15

    # Berechnung der Eigenenergien und Eigenzustände
    eigen_energien, normierte_ev, args = eigen_berechnungen(
        potential_von_A, x_min, x_max, anz_dim, h_eff, energie_grenze)
    # Skalierung und Plotten der Eigenzustände
    skalieren_und_plotten(eigen_energien, normierte_ev, args, potential_von_A)


if __name__ == "__main__":
    main()
    """
    a)
    Das Intervall [x_min, x_max] wurde so gewählt, dass alle Wellenfunktion 
    näherungsweise 0 außerhalb des Intervalls sind. Dies geschieht ab ca. 
    |x| >= 1,3 . Um Randeffekte bei der Berechnung zu vermeiden und um den 
    Abfall auf 0 der Wellenfunktionen klar darzustellen wurde |x| <= 2 als 
    Intervall gewählt. 
    Die Matrixgröße wurde mit 200 so groß gewählt, dass glatte Funktion zu 
    sehen sind, gleichzeitig aber so klein, dass keine übermäßige Rechenzeit 
    zu erwarten ist. So sind für eine Matrix der Dimension 20x20 deutliche 
    Kanten zu erkennen und für eine Dimension von 2000 ist ein Wartezeit im 
    Bereich von Sekunden zu erwarten.
    b) 
    i)
    Mittels folgendem Code wurde der Knotensatz überprüft: 
    
    from scipy.signal import find_peaks
    ...
    def skalieren_und_plotten(eigen_energien, normierte_ev, args, potential):
    ...
     for i in range(eigen_energien.size):
        maxima, _ = find_peaks(skalierte_ev[:, i])
        minima, _ = find_peaks(-skalierte_ev[:, i])
        num_extrema = len(maxima) + len(minima)
        print(num_extrema-1)
        
    Hier wird die Anzahl an Extrema - 1 ermittelt. Dies ist jedoch äquivalent 
    mit der Anzahl an Knoten für Lösungen der SGL.
    Das Ergebnis ist, wie nach Knotensatz erwartet, dass die n-te 
    Wellenfunktion genau n-1 Knoten aufweist. Mit bloßem Auge sind einige der 
    Extrema schwer zu erkennen. Für Wellenfunktionen mit Eigenenergien 
    niedriger als die Schwelle zwischen den beiden Mulden gilt, dass die 
    Extrema in einer der beiden Mulden deutlich höhere Maximalwerte haben, 
    als jene in der anderen Mulde. Dies ist als wesentlich höhere 
    Aufenthaltswahrscheinlichkeit in einer der beiden Mulden zu sehen. Diese 
    Wellenfunktionen sind abwechselnd in einer der beiden Mulden deutlich 
    stärker lokalisiert.
    ii)
    Für größere h_eff wird der mittlere Abstand zweier Energieniveaus größer, 
    die normierten Wellenfunktionen verbreitern sich und haben 
    dementsprechend geringere Extrema. Die Grundzustandsenergie steigt 
    ebenfalls für größere h_eff. Die Verbreiterung hat ebenfalls die Folge, 
    dass die Wellenfunktion weiter in den verbotenen Bereich eindringen.
    c) Für A=0 gibt es für Eigenenergien deutlich geringer als die Schwelle 
    zwischen den symmetrischen Mulden eine zweifache Entartung. Dies sind 
    keine exakten Entartungen, und je größer die Eigenenergien, desto größer 
    der Energieunterschied zwischen den Quasi-entarteten Zuständen. Bei 
    diesen Zuständen überlaggern sich die Wellenfunktionen in einer der 
    beiden Mulden, während sie sich in der anderen um ein Vorzeichen 
    unterscheiden. Diese numerisch ermittelten Lösungen erscheinen mir jedoch 
    unphysikalisch, da sie den Knotensatz verletzen. So gibt es zum Beispiel 
    keine Lösung ohne Knoten, also keinen Grundzustand.
    
    """
