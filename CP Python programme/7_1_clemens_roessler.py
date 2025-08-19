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


def erwartungswert(koeffizienten_array, observablen_array):
    """
    Berechnet den Erwartungswert einer Observablen.

    Args:
        koeffizienten_array (array): Array der Entwicklungskoeffizienten
        observablen_array (array): Array der Observablen

    Returns:
        float: Erwartungswert der Observablen
    """
    return np.dot(np.abs(koeffizienten_array) ** 2, observablen_array)


def skalierung_2(eigen_energien, eigen_funktionen):
    """
    Berechnet den Skalierungsfaktor der Eigenzustände ** 2 und der 
    Wellenfunktion ** 2 basierend auf den mittleren Abständenen der 
    Eigenenergien und dem Maxima der Eigenfunktionen.

    Args:
        eigen_energien (array): Array der Eigenenergien
        eigen_funktionen (array): Array der Eigenfunktionen

    Returns:
        float: Skalierungsfaktor der Eigenzustände ** 2 und der 
        Wellenfunktion ** 2
    """
    # Berechnung der Differenzen der Eigenenergien
    energie_diff = np.diff(eigen_energien)

    # Normquadrat der Eigenzustände
    eig_funk_2 = np.abs(eigen_funktionen) ** 2

    # Berechnung des maximalen Werts der normierten Eigenzustände
    max_eig_funk_2 = np.max(eig_funk_2)

    # Berechnung des mittleren Differenzwertes der Eigenenergien
    mean_energie_diff = np.mean(energie_diff)

    # Skalierungsfaktor der Eigenzustände ** 2 und der Wellenfunktion ** 2
    skalierungsfaktor = mean_energie_diff / (max_eig_funk_2)

    return skalierungsfaktor


def neues_Wellenpacket(event, ax, x_array, eigen_funktionen, eigen_energien,
                       zeiten, dx, skalierungsfaktor=0.01, delta_x=0.1,
                       h_eff=0.06, p_0=0):
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

        # Erzeugt ein gaußsches Wellenpaket  am Startpunkt
        phi_0 = gausssches_wellenpaket(x_array, x_0, delta_x, h_eff, p_0)

        # Ersellt array der Entwicklungskoeffizienten des Wellenpakets in die
        # Eigenfunktionen
        koeffizienten_array = entwicklungs_koeffizienten(
            phi_0, eigen_funktionen=eigen_funktionen, dx=dx)

        # Berechnet Energieerwartungswert
        energie_erwartungswert = erwartungswert(
            koeffizienten_array, eigen_energien)

        # Berechnet Zeitentwickeltes Wellenpacket
        phi_t_x = phi_t(zeiten, eigen_energien, eigen_funktionen,
                        koeffizienten_array, h_eff)

        # Berechnung und Ausgabe der Norm der Differenz zwischen dem
        # Gaußschen und dem Zeitentwickelten Wellenpacket bei t=0
        norm_differenz = np.sum(
            (np.abs(phi_0 - phi_t_x[0])) ** 2)
        print("""Die Norm der Differenz vom Gaußschen Wellenpacket und dem 
              Wellenpacket zur Zeit t_0 beträgt """ + str(norm_differenz))

        # Berechnung des Betragquadrats der Wellenfunktion
        phi_t_x_2 = np.abs(phi_t_x) ** 2

        # Skalierung und Verschiebung
        phi_t_x_2_skaliert = phi_t_x_2 * skalierungsfaktor \
            + energie_erwartungswert

        phi_plot = ax.plot(
            x_array, phi_t_x_2_skaliert[0])  # Anfangsplot
        for phi_x_2 in phi_t_x_2_skaliert[1:]:
            phi_plot[0].set_ydata(phi_x_2)  # Plotdaten aktualisieren,
            event.canvas.flush_events()  # und dynamisch
            event.canvas.draw()  # darstellen.


def main():
    """Hauptprogramm. Aufruf für verschiedene Parameter."""
    print(__doc__)

    # x-Intervall
    x_min = -1.5
    x_max = 1.5

    # gibt die Genauigkeit der Lösung der SGL der Wellenfunktion an
    anz_x_stellen = 200

    # effektives Plancksches Wirkungsquantum
    h_eff = 0.06

    # Potentialparameter
    A = 0.055

    # Breite des Wellenpakets
    delta_x = 0.1

    # Anfangsimpuls
    p_0 = 0

    # Grenze über welcher Eigenenergien nicht betrachtet werden
    energie_grenze = 0.15

    # Anzahl Zeitpunkte pro Zeiteintervall
    anz_zeit_per_intervall = 10

    # Zeit Anfang und Ende
    t_start = 0
    t_end = 10

    # Ausgabe der relevanten Parameter
    print(
        f"""
        Verwendete Parameter: Das betrachtete Intervall wurde von x =   
        {x_min} bis x = {x_max} gewählt. Die Genauigkeit der Lösung der SGL 
        der Wellenfunktion wurde als {anz_x_stellen} gewählt. Das effektives 
        Plancksche Wirkungsquantum h_eff wurde mit {h_eff} vorgegeben und der 
        Potentialparameter A wurde ebenfalls gegeben mit A = {A}. 
        Die Energiegrenze der Eigenenrgien wurde bei {energie_grenze} gesetzt.
        Die Breite des Wellenpakets war gegeben als {delta_x} und der 
        Anfangsimpuls als {p_0}. Die Anzahl der Zeitpunkte pro Zeiteintervall 
        wurde auf {anz_zeit_per_intervall} gesetz und das Zeitintervall 
        beginnt bei {t_start} und endet bei {t_end}.
        """)

    # Ausgabe der Benutzerführung
    print("""
          Sie sehen ein Diagramm welches das Potential in grauer Linie 
          darstellt. Die Eigenfunktionen sind auf Höhe der  
          zugöhrigen Eigenenergien eingezeichnet. Die Eigenenergien und das 
          Potential sind akurat gegenüber der Energieachse skaliert. Die 
          Eigenfunktionen hingegen sind alle mit einem Faktor so skaliert das 
          sie gut sichtbar sind. Sie können nun einen Startpunkt per Klick 
          auswählen um die Zeitentwicklung eines Gaußschen Wellenpacketes an 
          diesem Punkt zu starten.
          """)

    # Potentialfunktion mit festem A
    potential_von_A = functools.partial(doppelmulde, A=A)

    # zeiten
    zeiten = np.linspace(t_start, t_end, t_end * anz_zeit_per_intervall)

    # Erstellt Array der Ortskoordinaten
    x_array, dx = qm.diskretisierung(
        x_min, x_max, anz_x_stellen, retstep=True)

    # Berechnet Eigenenergien und Eigenfunktionen
    eigen_energien, eigen_funktionen = qm.diagonalisierung(
        h_eff, x_array, potential_von_A)

    # Berechnet den Skalierungsfaktor der Eigenzustände ** 2 und der
    # Wellenfunktion ** 2
    skalierungsfaktor = skalierung_2(eigen_energien, eigen_funktionen)

    # Erstellen des Plots
    fig, ax = plt.subplots()

    # Plottet Eigenfunktionen und Potential
    qm.plot_eigenfunktionen(
        ax, eigen_energien, eigen_funktionen, x_array, potential_von_A,
        betragsquadrat=True, Emax=energie_grenze, fak=skalierungsfaktor)

    # Bei Mausklick soll die Funktion neues_Wellenpacket aufgerufen werden
    klick_funktion = functools.partial(
        neues_Wellenpacket, ax=ax, x_array=x_array,
        eigen_funktionen=eigen_funktionen, eigen_energien=eigen_energien,
        zeiten=zeiten, dx=dx, skalierungsfaktor=skalierungsfaktor,
        delta_x=delta_x, h_eff=h_eff, p_0=p_0)

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
