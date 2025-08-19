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


def doppelmulde(x_array, A=0.055):
    """
    Berechnet das Doppelmuldenpotential an der Stelle x.

    Args:
        x_array (array): Array der Ortskoordinaten
        A (float): Parameter des Potentials

    Returns:
        array: Werte des Potentials an den Stellen x
    """
    return x_array ** 4 - x_array ** 2 + A * x_array


def hamilton_fkt(x, p, potential):
    """
    Berechnet die Hamilton-Funktion.

    Args:
        x (array): Array der Ortskoordinaten.
        p (array): Array der Impulskoordinaten.
        potential (function): Funktion für das Potential.

    Returns:
        array: Werte der Hamilton-Funktion.
    """
    return p**2 / 2 + potential(x)


def gausssches_wellenpaket(x, x_0=0, delta_x=0.1, h_eff=0.06, p_0=0):
    """
    Erzeugt ein gaußsches Wellenpaket mit gegebenen Parametern.

    Args:
        x (array): Array der Ortskoordinaten.
        x_0 (float): Zentrum des Wellenpakets in Ortsraum.
        delta_x (float): Breite des Wellenpakets.
        h_eff (float): Effektives Plancksches Wirkungsquantum.
        p_0 (float): Zentrum des Wellenpakets im Impulsraum.

    Returns:
        array: Werte des gaußschen Wellenpakets an den Ortskoordinaten.
    """
    return (1 / (delta_x * np.sqrt(np.pi)))**0.5 * np.exp(-(x - x_0)**2 / (2 * delta_x**2) + 1j * p_0 * x / h_eff)


def kohaerenter_zustand(x, x_0=0, p_0=0, h_eff=0.06):
    """
    Erzeugt einen kohärenten Zustand.

    Args:
        x (array): Array der Ortskoordinaten.
        x_0 (float): Zentrum des kohärenten Zustands in Ortsraum.
        p_0 (float): Zentrum des kohärenten Zustands im Impulsraum.
        h_eff (float): Effektives Plancksches Wirkungsquantum.

    Returns:
        array: Werte des kohärenten Zustands an den Ortskoordinaten.
    """
    delta_x = np.sqrt(h_eff / 2)
    return gausssches_wellenpaket(x, x_0, delta_x, h_eff, p_0)


def kohaerenter_zustand_tensor(eig_array, x_array, p_array, h_eff=0.06):
    """
    Erzeugt ein Tensor der kohärenten Zustände.

    Args:
        eig_array (array): Array der Eigenfunktionen.
        x_array (array): Array der Ortsgitterpunkte.
        p_array (array): Array der Impulsgitterpunkte.
        h_eff (float): Effektives Plancksches Wirkungsquantum.

    Returns:
        array: Tensor der kohärenten Zustände.
    """
    # Initialisierung des Tensors
    tensor = np.zeros((len(x_array), len(p_array),
                      len(eig_array)), dtype=complex)

    # Berechnung der kohärenten Zustände für jedes (x, p)-Paar
    for i, x_0 in enumerate(x_array):
        for j, p_0 in enumerate(p_array):
            tensor[i, j, :] = kohaerenter_zustand(eig_array, x_0, p_0, h_eff)

    return tensor


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
    zeit_energie_matrix = np.einsum("i,j", t, eigen_energien)

    # Exponentialfunktion auf alle Einträge der Matrix anwenden
    e_funktion = np.exp(-1j * zeit_energie_matrix / h_eff)

    # Berechnung der Zeitentwickelten Wellenfunktion per
    # Einsteinsummenkonvention (Kontratktion der Matrizen)
    phi_t_array = np.einsum("ij,j,kj", e_funktion, koeffizienten_array,
                            eigen_funktionen)

    return phi_t_array


def zeitentwickelte_husimi_darstellung(phi_0, t_array, dx, eigen_energien, eigen_funktionen, k_zustand_tensor, h_eff=0.06):
    """
    Berechnet die Husimi-Darstellung eines zeitentwickelten Wellenpakets.

    Args:
        phi_0 (array): Anfangswellenpaket in Ortsdarstellung.
        eig_array (array): Array der Eigenfunktionen.
        x_array (array): Array der Ortsgitterpunkte.
        p_array (array): Array der Impulsgitterpunkte.
        t_array (array): Array der Zeiten.
        potential (function): Funktion für das Potential.
        dx (float): Gitterabstand in der Ortsdarstellung.
        eigen_energien (array): Array der Eigenenergien.
        eigen_funktionen (array): Array der Eigenfunktionen.
        k_zustand_tensor (tensor): Tensor der kohärenten Zustände.
        h_eff (float): Effektives Plancksches Wirkungsquantum.

    Returns:
        array: Husimi-Darstellungen des Wellenpakets für verschiedene Zeiten.
    """

    # Erstellt array der Entwicklungskoeffizienten des Wellenpakets in die
    # Eigenfunktionen
    koeffizienten_array = entwicklungs_koeffizienten(
        phi_0, eigen_funktionen=eigen_funktionen, dx=dx)

    # Berechnet Zeitentwickeltes Wellenpacket
    phi_t_x = phi_t(t_array, eigen_energien, eigen_funktionen,
                    koeffizienten_array, h_eff)

    # reskalierung für mittelpunktintegration
    randindexe = [0, -1]
    k_zustand_tensor[randindexe] *= 1/2
    stammfunktion = np.einsum(
        ",jki,li->lkj", dx, np.conj(k_zustand_tensor), phi_t_x)

    husimi_matrix_array = np.abs(stammfunktion) ** 2 / h_eff
    return husimi_matrix_array


def neues_Wellenpacket(event, ax, eig_array, x_array, p_array, t_array, dx, eigen_energien, eigen_funktionen, k_zustand_tensor, h_eff=0.06):
    """
    Berechnet die Zeitentwicklung eines neuen Wellenpakets basierend auf einem Mausklickereignis und stellt es dar.

    Args:
        event: Mausklickereignis.
        ax: Achsenobjekt für das Plotten.
        eig_array (array): Array der Eigenfunktionen.
        x_array (array): Array der Ortsgitterpunkte.
        p_array (array): Array der Impulsgitterpunkte.
        t_array (array): Array der Zeiten.
        potential (function): Funktion für das Potential.
        dx (float): Gitterabstand in der Ortsdarstellung.
        eigen_energien (array): Array der Eigenenergien.
        eigen_funktionen (array): Array der Eigenfunktionen.
        k_zustand_tensor (tensor): Tensor der kohärenten Zustände.
        h_eff (float): Effektives Plancksches Wirkungsquantum.
    """

    # Teste, ob Klick mit linker Maustaste im Plotfenster ax erfolgt
    # sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes == ax and mode == '':
        # Extrahieren der Koordinaten des Mausklicks
        x_0 = event.xdata
        p_0 = event.ydata

        # Erzeugen eines neuen kohärenten Zustands am Mausklickpunkt
        phi_0 = kohaerenter_zustand(eig_array, x_0, p_0, h_eff)

        # Berechnung der Husimi-Darstellung des neuen Wellenpakets
        hus_matrix_array = zeitentwickelte_husimi_darstellung(
            phi_0, t_array, dx=dx, eigen_energien=eigen_energien, eigen_funktionen=eigen_funktionen, k_zustand_tensor=k_zustand_tensor, h_eff=h_eff)

        # Darstellung der Husimi-Darstellung
        hus_img_plot = ax.imshow(hus_matrix_array[0], aspect="auto",
                                 extent=(x_array[0], x_array[-1], p_array[0], p_array[-1]))  # Anfangsplot

        for hus_matrix in hus_matrix_array[1:]:
            hus_img_plot.set_data(hus_matrix)  # Plotdaten aktualisieren,
            event.canvas.flush_events()  # und dynamisch
            event.canvas.draw()  # darstellen.

        # Entfernen des plots nach Zeitentwicklung
        hus_img_plot.remove()


def main():
    """
    Hauptfunktion des Programms. Initialisiert die Parameter, berechnet die
    notwendigen Größen und erstellt den interaktiven Plot.
    """
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
    anz_x = 150

    # bestimmt Genauigkeit der Rechnung
    N_eig = 100

    # gibt die Anzahl der gewählten p-Punkte an
    anz_p = 152

    # effektives Plancksches Wirkungsquantum
    h_eff = 0.06

    # Potentialparameter
    A = 0.055

    # Anzahl Zeitpunkte pro Zeiteintervall
    anz_zeit_per_intervall = 3

    # Zeit Anfang und Ende
    t_start = 0
    t_end = 12

    # Niveaus fuer Konturplot
    contour_werte = [-0.25, -0.2, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 2.0]

    # Potentialfunktion mit festem A
    potential_von_A = functools.partial(doppelmulde, A=A)

    # Zeiten
    t_array = np.linspace(t_start, t_end, t_end * anz_zeit_per_intervall)

    # Erstellt Array der Impulse
    p_array = np.linspace(*p_intervall, anz_p)

    # Erstellt Array der Ortskoordinaten für die Plotdarstellung
    x_array = np.linspace(x_min, x_max, anz_x)

    # Erstellt Array der Ortskoordinaten für die Eigenfunktionen
    eig_array, dx = qm.diskretisierung(
        x_min, x_max, N_eig, retstep=True)

    # Berechnet Eigenenergien und Eigenfunktionen
    eigen_energien, eigen_funktionen = qm.diagonalisierung(
        h_eff, eig_array, potential_von_A)

    # Berechnet einen Tensor von kohaerenten Zuständen
    k_zustand_tensor = kohaerenter_zustand_tensor(
        eig_array, x_array, p_array, h_eff)

    # Erstelle 2D meshgrid aus 1D arrays in (x, p)
    x2, p2 = np.meshgrid(x_array, p_array)

    # Berechne Hamiltonfunktion
    hamilton = hamilton_fkt(x2, p2, potential_von_A)

    # Erstellen des Plots
    fig, ax = plt.subplots()

    # Achsenbereiche setzen
    ax.set_xlim(*x_intervall)
    ax.set_ylim(*p_intervall)

    # Zeichne Konturlinien und erstelle label
    kontur_linien = ax.contour(x_array, p_array, hamilton, contour_werte,
                               colors='k', linestyles='solid')
    ax.clabel(kontur_linien, inline=1, fontsize=10)

    # Bei Mausklick soll die Funktion neues_Wellenpacket aufgerufen werden
    klick_funktion = functools.partial(
        neues_Wellenpacket, ax=ax, eig_array=eig_array, x_array=x_array, p_array=p_array, t_array=t_array, dx=dx, eigen_energien=eigen_energien, eigen_funktionen=eigen_funktionen, k_zustand_tensor=k_zustand_tensor, h_eff=h_eff)

    # übergibt das mausklick-event an die klick_funktion
    fig.canvas.mpl_connect("button_press_event", klick_funktion)

    # zeichnet plot
    plt.show()


if __name__ == "__main__":
    main()
