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




from scipy.signal import find_peaks
import functools
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np
def doppelmulde(x, A=0.055):
    return x ** 4 - x ** 2 + A * x


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


def skalieren_und_plotten(eigen_energien, normierte_ev, args, potential):
    energie_diff = np.diff(eigen_energien)
    max_energie = np.max(np.abs(normierte_ev))
    mean_energie_diff = np.mean(energie_diff)

    skalierte_ev = normierte_ev * mean_energie_diff / (2 * max_energie)
    potential_kurve = potential(args)

    x_achsengrenze = [np.min(args), np.max(args)]
    E_achsengrenze = [np.min(potential_kurve),
                      np.max(eigen_energien) + mean_energie_diff / 2]

    # Erstellen der Plots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Achsenbereiche setzen
    ax.set_xlim(*x_achsengrenze)
    ax.set_ylim(*E_achsengrenze)
    # Achsen labeln
    ax.set_xlabel("Stelle x")
    ax.set_ylabel("Energie E")
    # Diagrammtitel
    ax.set_title("Teilchen in Doppelmuldenpotential")

    for i in range(eigen_energien.size):
        ax.plot(args, skalierte_ev[:, i] + eigen_energien[i],
                ls="-", label=f"Eigenzustand {i+1}")
        maxima, _ = find_peaks(skalierte_ev[:, i])
        minima, _ = find_peaks(-skalierte_ev[:, i])
        num_extrema = len(maxima) + len(minima)
        print(num_extrema-1)
    ax.plot(args, potential_kurve, ls="--",
            color="black", label="Potentialkurve")

    # Legende hinzufügen
    ax.legend(loc="lower right")

    # plot zeichnen
    plt.show()


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)

    # Ausgabe der relevanten Parameter

    # Ausgabe der Benutzerführung

    # Parameter
    x_min = -2
    x_max = 2
    anz_dim = 200
    h_eff = 0.06
    A = 0.055 * 0.15
    potential_von_A = functools.partial(doppelmulde, A=A)
    energie_grenze = 0.15

    eigen_energien, normierte_ev, args = eigen_berechnungen(
        potential_von_A, x_min, x_max, anz_dim, h_eff, energie_grenze)
    skalieren_und_plotten(eigen_energien, normierte_ev,
                          args, potential_von_A)


if __name__ == "__main__":
    main()
