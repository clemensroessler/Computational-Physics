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




import functools
import quantenmechanik as qm
import matplotlib.pyplot as plt
import numpy as np
def doppelmulde(x, A=0.055):
    """Funktion zur Berechnung des Doppelmuldenpotentials an Stelle x für
    Parameter A."""
    return x ** 4 - x ** 2 + A * x


def gausssches_wellenpaket_0(x, x_0=0, dx=0.1, h_eff=0.06, p_0=0):
    return np.exp(-(x - x_0) ** 2 / (4 * dx ** 2)) \
        / (2 * np.pi * dx ** 2) ** (1/4) * np.exp(1j / h_eff * p_0 * x)


def entwicklungs_koeffizienten(phi_0, eigen_funktionen, delta_x):
    return delta_x * np.conjugate(np.transpose(eigen_funktionen)) @ phi_0


def phi_t(t, eigen_energien, eigen_funktionen, koeffizienten_array, h_eff=0.06):
    # stellt eine 2 dimensionale Matrix wobei die Einträge die Zeiten multipliziert mit den Energien sind
    zeit_energie_matrix = np.tensordot(t, eigen_energien, axes=0)
    # die e_funktion wirkt auf alle Einträge der Matrix einzeln
    e_funktion = np.exp(-1j * zeit_energie_matrix / h_eff)
    # der n-te Koeffizient wird mit den e_funktionen der n-ten Eigenenergie multipliziert
    skalierte_e_funktion = np.multiply(e_funktion, koeffizienten_array)
    # die n-ten skalierten e-funktionen werden mit den n-ten eigenfunktionen multipliziert und dannach werden diese Ergebnisse für alle n aufaddiert, sodass man eine Matrix in abhängigkeit von Ort und Zeit erhält
    phi_t_array = np.tensordot(skalierte_e_funktion.T,
                               eigen_funktionen.T, axes=(0, 0))
    return phi_t_array


def erwartungswert(koeffizienten_array, observablen_array):
    return np.dot(np.abs(koeffizienten_array) ** 2, observablen_array)


def neues_Wellenpacket(event, ax, x_min, x_max, anz_dim, potential, zeiten, dx=0.1, h_eff=0.06, p_0=0):
    """Iteriert und plottet neue Trajektorie der Standardabbildung.

    Der Anfangspunkt wird durch Mausklick festgelegt.

    Hierbei enthaelt event die Informationen ueber den Mausklick,
    ax den Plotbereich.
    Fuer die Standardabbildung mit Kickstaerke K werden anz Iterationen
    durchgefuehrt und dargestellt.
    """
    # Teste, ob Klick mit linker Maustaste im Plotfenster ax erfolgt
    # sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax and mode == '':
        x_0 = event.xdata
        x_array, delta_x = qm.diskretisierung(
            x_min, x_max, anz_dim, retstep=True)
        phi_0 = gausssches_wellenpaket_0(x_array, x_0, dx, h_eff, p_0)
        eigen_energien, eigen_funktionen = qm.diagonalisierung(
            h_eff, x_array, potential)
        koeffizienten_array = entwicklungs_koeffizienten(
            phi_0, eigen_funktionen=eigen_funktionen, delta_x=delta_x)
        phi_t_x = phi_t(zeiten, eigen_energien, eigen_funktionen,
                        koeffizienten_array, h_eff)

        phi_plot = ax.plot(x_array, phi_0)  # Anfangsplot
        for phi_x in phi_t_x:
            phi_plot[0].set_ydata(phi_x)  # Plotdaten aktualisieren,
            event.canvas.flush_events()  # und dynamisch
            event.canvas.draw()  # darstellen.


def main():
    """Hauptprogramm. Aufruf für verschiedene Parameter."""
    print(__doc__)

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
    # Anzahl Zeitpunkte
    anz_zeit = 205
    # Zeit Anfang und Ende
    t_start = 0
    t_end = 20
    # zeiten
    zeiten = np.linspace(t_start, t_end, anz_zeit)

    x, delta_x = qm.diskretisierung(x_min, x_max, anz_dim, retstep=True)
    # eigenfunktionen hat form : n_1, n_2, ..
    #                           x_2, ...
    #                           x_3, ...
    eigen_energien, eigen_funktionen = qm.diagonalisierung(
        h_eff, x, potential_von_A)
    phi_0 = gausssches_wellenpaket_0(
        x, h_eff=h_eff)
    koeffizienten_array = entwicklungs_koeffizienten(
        phi_0, eigen_funktionen=eigen_funktionen, delta_x=delta_x)
    # print(np.sum(np.abs(koeffizienten_array) ** 2))
    phi = phi_t(zeiten, eigen_energien, eigen_funktionen,
                koeffizienten_array, h_eff)
    """
    norm_differenz = np.sum(
        (phi_0 - phi[0]) ** 2)
    print(norm_differenz)
    """
    print(erwartungswert(koeffizienten_array, eigen_energien))
    # Erstellen der Plots
    fig = plt.figure(figsize=(14, 8))

    # SIR
    ax = fig.add_subplot(1, 1, 1)
    # Achsenbereiche setzen
    # ax.set_xlim(*t_achsengrenze)
    # ax.set_ylim(*N_achsengrenze)
    # Achsen labeln
    ax.set_xlabel("Zeit t")
    ax.set_ylabel("Anteil der Bevölkerung N_t/N")
    # Diagrammtitel
    ax.set_title("SIR-Modell")
    ax.plot(x, phi[20], c='r', ls='-', label="S normal")
    plt.show()


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
