"""Differentialgleichungen II: SIR-Modell
Das Programm  ermittelt die Lösung des Differentialgleichungssystems des 
SIR-Modells und zeichnet die S, I, und R-Anteile der Bevölkerung mit und ohne 
Lockdown, sowie die maximale Belastungsgrenze des Gesundheitssystems in zwei 
Plots ein.
"""
"""
Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs
Code die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die 
nächste Zeile kommentiert/entfernt werden.
"""


# %matplotlib ipympl




import math
from scipy.special import hermite
import functools
from scipy.linalg import eigh
from scipy.integrate import odeint          # Integrationsroutine fuer DGL
import matplotlib.pyplot as plt
import numpy as np
def potential(x, A=0.055):
    return x ** 4 - x ** 2 + A * x


def harmonisch(x):
    return x ** 2 / 2


def eigen_berechnungen(potential, x_min=-3, x_max=3, anz_dim=200, h_eff=0.06, energie_grenze=0.15):
    delta_x = (x_max - x_min) / (anz_dim+1)
    args = x_min + (np.arange(anz_dim) + 1) * delta_x
    V_array = potential(args)
    z_konstante = h_eff ** 2 / (2 * delta_x ** 2)
    z_array = np.ones(anz_dim - 1) * z_konstante
    eigenmatrix = np.diag(V_array + 2 * z_konstante) + \
        np.diag(-z_array, k=-1) + np.diag(-z_array, k=1)
    eigen_energien, eigenvektoren = eigh(
        eigenmatrix, subset_by_value=[-np.inf, energie_grenze])
    normierte_ev = eigenvektoren * delta_x ** (-1 / 2)
    return eigen_energien, normierte_ev, args


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)

    # Ausgabe der relevanten Parameter

    # Ausgabe der Benutzerführung

    # Parameter
    x_min = -2
    x_max = 2
    x_achsengrenze = [x_min, x_max]
    anz_dim = 200
    h_eff = 0.06
    A = 0.055
    potential_von_A = functools.partial(potential, A=A)
    energie_grenze = 0.15

    eigen_energien, normierte_ev, args = eigen_berechnungen(
        potential_von_A, x_min, x_max, anz_dim, h_eff, energie_grenze)
    min_energie_diff = np.min(np.diff(eigen_energien))
    normierte_ev = normierte_ev * min_energie_diff / 2
    anz_energien = eigen_energien.size
    print(eigen_energien)

    E_achsengrenze = [np.min(eigen_energien) * 1.5,
                      np.max(eigen_energien) * 1.5]

    # Erstellen der Plots
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    # Achsenbereiche setzen
    ax.set_xlim(*x_achsengrenze)
    ax.set_ylim(*E_achsengrenze)
    # Achsen labeln
    ax.set_xlabel("Stelle x")
    ax.set_ylabel("Energie E")
    # Diagrammtitel
    ax.set_title("Teilchen in Doppelmuldenpotential")

    # Plot mit Farben und Beschriftungen fuer Legende fuer SIR
    # ohne Lockdown
    for i in range(anz_energien):
        ax.plot(args, normierte_ev[:, i] +
                eigen_energien[i], ls='-')
        """
        h = hermite(i)
        f_test = 1 / (np.pi ** (1/4) * (2**i * math.factorial(i))
                      ** (1/2)) * np.exp(-1/2 * args ** 2) * h(args)
        ax.plot(args, f_test + eigen_energien[i], ls='--')
        """
    ax.plot(args, potential_von_A(args), ls='--')
    # Legende
    ax.legend(loc='lower left')
    # plot zeichnen
    plt.show()


if __name__ == "__main__":
    main()

"""
a)
i)
Für sehr hohe b wird das Maximum des I-Wertes noch vor dem Lockdown erreicht. 
Bei b=3 z.B. sind nach ca. 8 Tagen 85% der Bevölkerung infiziert. Für b=1.5 
dauert es schon 15 Tage bis das Maximum von 75% erreicht ist. Für b=1 wird 
das Maximum nach ca. 23 Tagen erreicht. Dementsprechend ist für diesen 
b-Wert, sowie niedrigere b-Werte ein Unterschied zwischen dem Verlauf mit und 
ohne Lockdown zu sehen. So liegt der Maximalwert mit Lockdown bei 46% und 
ohne Lockdown bei 67%. Die Zeit nach dem das Maximum erreicht Unterscheidet 
sich hier noch nicht großartig. Bei einem b Wert von 0.8 lässt sich hier 
schon ein deutlicher Unterschied sowohl in der Zeit als auch in der Höhe des 
Maximums sehen. So wird ohne Lockdown das Maximum nach 30 Tagen erreicht und 
mit Lockdown nach 50 Tagen. Die Maxima liegen hier bei ca. 50% und 23%.
Für b=0.5 sind die Unterschiede noch ersichtlicher, da ohne Lockdown nach 50 
Tagen das Maximum erreicht wird, mit Lockdown jedoch erst nach 115. Die 
Maxima liegen immer noch über der Belastungsgrenze mit 45% und 23% respektiv.
Unter b=0.3 kommt es mit Lockdown zu keiner ersichtlichen Infizierung. Ohne 
Lockdown wird das Maximum nach 95 Tagen mit 30% erreicht. Für b<=0.18 kommt 
es selbst ohne Lockdown zu keiner erkennbaren Rate an infizierten. 
ii) Ein größeres b sorgt für einen steileren Anstieg und eine Verschmälerung 
des Maximas. Ein kleines gamma (hier c genannt) erhöht den Anstieg der 
I-Kurve und senkt drastisch die Rate in welcher Die Kurve abfällt. Ein 
erhöhen von b und c bei konstanten Verhältnis b/c sorgt für ein steileres und 
schmäleres Maximum, welches eher erreicht wird.
b) Der Wert von R_0 bei welchem sich die Belastungsgrenze mit der 
Maxmia-Kurve schneidet liegt bei 2,08 bis 2,09.
c) Der Lockdown und seine Auswirkungen wurden schon für verschiedene b-Werte 
in a) diskutiert. Für den konkreten Wert von b=0.5 lässt sich sehen, dass 
erst nach Tag 60 die Rate der Infizierten signifikant steigt. Das entstehende 
Maximum der I-Kurve ist jedoch mit 23% deutlich niedrieger als Maximum ohne 
Lockdown bei 48%. Dies liegt an der Lockerung der Maßnahmen. Bei einem 
Aufheben der Maßnahmen hätte man stattdessen nur das Maximum verzögert, die 
Höhe jedoch nicht verändert.
"""
