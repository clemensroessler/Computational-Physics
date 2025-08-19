"""Quantenmechanik von 1D-Potentialen II: Zeitentwicklung
Das Programm betrachtet ein Gaußsches Wellenpaket im asymmetrischen
Doppelmuldenpotential und bestimmt dessen Zeitentwicklung mit klickbarem
Anfagspunkt und stellt diese dann grafisch zusammen mit dem Potential und den
Eigenfunktionen dar.
"""

"""
Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs
Code, die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die
nächste Zeile kommentiert/entfernt werden.
"""

import numpy as np
import matplotlib.pyplot as plt
import quantenmechanik as qm
import functools
%matplotlib ipympl


def doppelmulde(x, A=0.055):
    """
    Berechnet das Doppelmuldenpotential an der Stelle x.

    Args:
        x (array): Array der Ortskoordinaten
        A (float): Parameter des Potentials

    Returns:
        array: Werte des Potentials an den Stellen x
    """
    return x ** 4 - x ** 2 + A * x


def hamilton_fkt(x, p, potential):
    return p ** 2 / 2 + potential(x)


def gausssches_wellenpaket(x, x_0=0, delta_x=0.1, h_eff=0.06, p_0=0):
    """
    Erzeugt ein gaußsches Wellenpaket mit den gegebenen Parametern.

    Args:
        x (array): Array der Ortskoordinaten
        x_0 (float): Anfangsort des Wellenpakets
        delta_x (float): Breite des Wellenpakets
        h_eff (float): Effektives Plancksches Wirkungsquantum
        p_0 (float): Anfangsimpuls

    Returns:
        array: Werte des Wellenpakets an den Stellen x
    """
    return np.exp(-(x - x_0) ** 2 / (4 * delta_x ** 2)) \
        / (2 * np.pi * delta_x ** 2) ** (1/4) * np.exp(1j / h_eff * p_0 * x)


def kohaerenter_zustand(x, x_0=0, p_0=0, h_eff=0.06):
    return np.exp(-(x - x_0) ** 2 / (2 * h_eff)) \
        / (np.pi * h_eff) ** (1/4) * np.exp(1j / h_eff * p_0 * x)


def kohaerenter_zustand_tensor(x_array, p_array, h_eff=0.06):
    faktor = (1 / (np.pi * h_eff)) ** (1 / 4)
    x_matrix = np.tensordot(x_array, np.ones(x_array.size), axes=0)
    x_x_matrix = x_matrix - x_matrix.T
    x_x_exp_matrix = np.exp(x_x_matrix ** 2 / (-2 * h_eff))
    # p steht eigentlich für den index k, jedoch muss für die einsum funktion mit i begonnen werden
    p_x_matrix = np.einsum("i,j", p_array, x_array)
    p_x_exp_matrix = np.exp(1j / h_eff * p_x_matrix)
    k_zustand_tensor = np.einsum(
        ",ij,kj->jik", faktor, x_x_exp_matrix, p_x_exp_matrix)
    return k_zustand_tensor


def entwicklungs_koeffizienten(phi_0, eigen_funktionen, dx):
    """
    Berechnet die Entwicklungskoeffizienten des Wellenpakets in die
    Eigenfunktionen.

    Args:
        phi_0 (array): Anfangswellenpaket
        eigen_funktionen (array): Array der Eigenfunktionen
        dx (float): Schrittweite des Ortsarrays

    Returns:
        array: Entwicklungskoeffizienten des Wellenpakets
    """
    return dx * np.conjugate(np.transpose(eigen_funktionen)) @ phi_0


def phi_t(t, eigen_energien, eigen_funktionen, koeffizienten_array,
          h_eff=0.06):
    """
    Berechnet das zeitentwickelte Wellenpaket.

    Args:
        t (array): Array der Zeiten
        eigen_energien (array): Array der Eigenenergien
        eigen_funktionen (array): Array der Eigenfunktionen
        koeffizienten_array (array): Array der Entwicklungskoeffizienten
        h_eff (float): Effektives Plancksches Wirkungsquantum

    Returns:
        array: Zeitentwickeltes Wellenpaket als Funktion von Ort und Zeit
    """

    # Matrix der Zeiten multipliziert mit den Eigenenergien
    zeit_energie_matrix = np.tensordot(t, eigen_energien, axes=0)

    # Exponentialfunktion auf alle Einträge der Matrix anwenden
    e_funktion = np.exp(-1j * zeit_energie_matrix / h_eff)

    # der n-te Koeffizient wird mit den e_funktionen der n-ten Eigenenergie
    # multipliziert
    skalierte_e_funktion = np.multiply(e_funktion, koeffizienten_array)

    # die n-ten skalierten e-funktionen werden mit den n-ten eigenfunktionen
    # multipliziert und dannach werden diese Ergebnisse für alle n
    # aufaddiert, sodass man eine Matrix in abhängigkeit von Ort und Zeit
    # erhält
    phi_t_array = np.tensordot(skalierte_e_funktion.T,
                               eigen_funktionen.T, axes=(0, 0))

    return phi_t_array


def zeitentwickelte_husimi_darstellung(phi_0, x_array, p_array, t_array, dx, potential, h_eff=0.06):

    # Berechnet Eigenenergien und Eigenfunktionen
    eigen_energien, eigen_funktionen = qm.diagonalisierung(
        h_eff, x_array, potential)

    # Ersellt array der Entwicklungskoeffizienten des Wellenpakets in die
    # Eigenfunktionen
    koeffizienten_array = entwicklungs_koeffizienten(
        phi_0, eigen_funktionen=eigen_funktionen, dx=dx)

    # Berechnet Zeitentwickeltes Wellenpacket
    phi_t_x = phi_t(t_array, eigen_energien, eigen_funktionen,
                    koeffizienten_array, h_eff)

    k_zustand_tensor = kohaerenter_zustand_tensor(x_array, p_array, h_eff)

    integrand = np.einsum("jik,li->ljki", np.conj(k_zustand_tensor), phi_t_x)

    stammfunktion = np.sum(dx * integrand, axis=1) - \
        dx / 2 * (integrand[:, 0, :, :] + integrand[:, -1, :, :])

    husimi_matrix_array = np.abs(stammfunktion) ** 2 / h_eff

    """
    husimi_matrix_array = np.zeros(
        shape=(t_array.size, x_array.size, p_array.size))
    for i in range(t_array.size):
        husimi_matrix = husimi_darstellung(
            phi_t_x[i], x_array, p_array, dx, h_eff)
        husimi_matrix_array[i] = husimi_matrix
    """
    return husimi_matrix_array


def neues_Wellenpacket(event, ax, x_array, p_array, t_array, dx, potential, h_eff=0.06):
    """
    Die Funktion berechnet ausgehend vom gegebenen Startpunkt die 
    Zeitentwicklug eines gausschen Wellenpacketes und plottet diese in den 
    übergenen Plot.

    Args:
    event: Matplotlib-Ereignisobjekt, das den Mausklick enthält
    ax: Achsenobjekt für das Plotten
    x_array (array): Array der Ortskoordinaten
    eigen_funktionen (array): Array der Eigenfunktionen
    eigen_energien (array): Array der Eigenenergien
    zeiten (array): Zeiten für die Zeitentwicklung
    dx (float): Schrittweite des Ortsarrays
    skalierungsfaktor (float): Skalierungsfaktor der Eigenzustände ** 2 und 
    der Wellenfunktion ** 2
    delta_x (float): Breite des Wellenpakets
    h_eff (float): Effektives Plancksches Wirkungsquantum
    p_0 (float): Anfangsimpuls


    """

    # Teste, ob Klick mit linker Maustaste im Plotfenster ax erfolgt
    # sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax and mode == '':
        # Startpunkt x0 wird durch Klick bestimmt
        x_0 = event.xdata
        p_0 = event.ydata

        # Erzeugt ein gaußsches Wellenpaket  am Startpunkt
        phi_0 = kohaerenter_zustand(x_array, x_0, p_0, h_eff)

        hus_matrix_array = zeitentwickelte_husimi_darstellung(
            phi_0, x_array, p_array, t_array, dx, potential, h_eff)
        hus_img_plot = ax.imshow(hus_matrix_array[0], aspect="auto",
                                 extent=(x_array[0], x_array[-1], p_array[0], p_array[-1]))  # Anfangsplot
        for hus_matrix in hus_matrix_array[1:]:
            hus_img_plot.set_data(hus_matrix)  # Plotdaten aktualisieren,
            event.canvas.flush_events()  # und dynamisch
            event.canvas.draw()  # darstellen.


def main():
    """Hauptprogramm. Aufruf für verschiedene Parameter."""
    print(__doc__)

    # x-Intervall
    x_min = -1.5
    x_max = 1.5
    x_intervall = [x_min, x_max]

    # p-Intervall
    p_min = -2
    p_max = 2
    p_intervall = [p_min, p_max]

    # gibt die Anzahl der gewählten x-Punkte an
    anz_x = 300

    # gibt die Anzahl der gewählten p-Punkte an
    anz_p = 152

    # effektives Plancksches Wirkungsquantum
    h_eff = 0.06

    # Potentialparameter
    A = 0.055

    # Anzahl Zeitpunkte pro Zeiteintervall
    anz_zeit_per_intervall = 1

    # Zeit Anfang und Ende
    t_start = 0
    t_end = 2

    # Niveaus fuer Konturplot
    contour_werte = [-0.25, -0.2, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 2.0]

    # Potentialfunktion mit festem A
    potential_von_A = functools.partial(doppelmulde, A=A)

    # zeiten
    t_array = np.linspace(t_start, t_end, t_end * anz_zeit_per_intervall)

    # Erstellt Array der Ortskoordinaten
    x_array, dx = qm.diskretisierung(
        *x_intervall, anz_x, retstep=True)

    # Erstellt Array der Impulse
    p_array = np.linspace(*p_intervall, anz_p)

    x2, p2 = np.meshgrid(x_array, p_array)
    # Berechne Hamiltonfunktion
    hamilton = hamilton_fkt(x2, p2, potential_von_A)

    # Erstellen des Plots
    fig, ax = plt.subplots()

    # Achsenbereiche setzen
    ax.set_xlim(*x_intervall)
    ax.set_ylim(*p_intervall)
    kontur_linien = ax.contour(x_array, p_array, hamilton, contour_werte,
                               colors='k', linestyles='solid')
    ax.clabel(kontur_linien, inline=1, fontsize=10)

    # Bei Mausklick soll die Funktion neues_Wellenpacket aufgerufen werden
    klick_funktion = functools.partial(
        neues_Wellenpacket, ax=ax, x_array=x_array, p_array=p_array, t_array=t_array, dx=dx, potential=potential_von_A, h_eff=h_eff)

    # übergibt das mausklick-event an die klick_funktion
    fig.canvas.mpl_connect("button_press_event", klick_funktion)

    # zeichnet plot
    plt.show()


if __name__ == "__main__":
    main()
    """
    a)
    i) Beim Start des Wellenpaketes im Minimum des Potentials beobachtet man,
    dass das Betragsquadrat der Wellenfunktion (BqdW) im zeitlichen Mittel
    dem Grundzustand sehr ähnelt. Die Gaußsche Form bleibt erhalten bzw. es
    entstehen keine neuen Maxima oder Minima. Das BqdW bleibt auf einem
    beschränkten Raum in der Potentialmulde lokalisiert.
    ii)  Beim Start des Wellenpaketes im Maximum des Potentials beobachtet
    man eine unmittelbare Verbreiterung des BqdW. Zwei Maxima der
    Aufenthaltswahrscheinlichkeit breiten sich annähernd symmetrisch in
    positive und negative x Richtung aus. Nach dem diese Maxima an der
    Potentialwand auftreffen, prallen sie förmlich daran ab und werden in die
    Mitte zurückreflektiert. Nach wenigen Reflektionen beobachtet man ein
    chaotisches Verhalten mit annähernder Gleichverteilung des BqdW.
    b)
    i) Beim Start des Wellenpaketes im Minimum für p_0=0.3, beobachtet man
    zuerst eine deutlich höhere Eigenenergie bei gleichem Startpunkt x_0.
    Durch diese höhere Energie hat das Teilchen mehr Spielraum und weicht
    stärker von der Gaußschen Form ab. Das BqdW kehrt jedoch in
    unregelmäßigen Abständen in diese Form zurück. Ein weiterer Effekt der
    höheren Energie ist, dass man wesentlich mehr tunnelung von Teilen des
    BqdW sieht.
    ii) Beim Start des Wellenpaketes im Maximum für p_0=0.3, beobachtet man
    ebenfalls eine deutlich höhere Eigenenergie bei gleichem Startpunkt x_0.
    Die Effekte dieser höheren Energie jedoch keinen großen Einfluss auf das
    qualitative Verhalten des BqdW im Vergleich zu p_0=0.
    c) Für den Fall des symmetrischen Doppelmuldenpotentials (also A=0)
    stellt man fest, dass ein Wellenpacket, welches in einem Minima startet
    langsam in das andere Minima tunnelt. Dabei ist zuerst hauptsächlich im
    Startminima lokalisiert. Nach ca. 5000 Zeiteinheiten sieht man eine
    annäherende Gleichverteilung des BqdW auf die zwei Potentialminima. Bei
    ca. t= 10000 ist die Welle dann näherungsweise komplett in dem anderen
    Minima lokalisiert. Das BqdW lässt sich als Überlagerung zweier
    Gaußwellen in den Minima beschreiben.
    """
