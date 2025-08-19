"""Aufgabe 1: Standardabbildung

Beim Klicken innerhalb des Plotfensters wird, fuer einen mittels der linken Maustas- ¨
te vorgegebenen Startpunkt, die Standardabbildung des gekickten Rotors innerhalb des Phasenraums dargestellt.

"""

""" Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs Code 
die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die nächste Zeile kommentiert/entfernt werden."""

# %matplotlib ipympl




import functools
import numpy as np
import matplotlib.pyplot as plt
def iterationsschritt_standardabbildung(p_n, phi_n, kickstärke=2.6):
    """Berechnet den (n+1)ten Impuls und (n+1)ten Winkel der Standardabbildung des gekickten Rotors ausgehend von den n-ten
    Werten, sowie einer variablen Kickstärke"""
    # initialisiert den (n+1)ten Winkel und berechnet ihn mittels Standardabbildung
    phi_n1 = (phi_n + p_n) % (2 * np.pi)
    # initialisiert den (n+1)ten Impuls und berechnet ihn mittels Standardabbildung
    p_n1 = ((kickstärke * np.sin(phi_n1) + p_n)+np.pi) % (2*np.pi) - np.pi
    return phi_n1, p_n1


def gesamt_iteration_standardabbildung(offset=np.zeros(2), kickstärke=2.6, anz_iterationen=1000):
    """Berechnet Punkte im Phasenraum für die Standardabbildung des gekickten Rotors,
    mit variablem Startpunkt, variabler Kickstärke und Anzahl an Iterationsschritten.
    """
    # initialisiert den array der Winkelwerte
    phi_array = np.zeros(anz_iterationen+1)
    # initialisiert den array der Impulswerte
    p_array = np.zeros(anz_iterationen+1)
    phi_array[0] = offset[0]  # übergibt die Startwerte in die Arrays
    p_array[0] = offset[1]

    for i in range(anz_iterationen):
        """Führt die Iteration über beide arrays aus"""
        phi_array[i+1], p_array[i+1] = iterationsschritt_standardabbildung(
            phi_array[i], p_array[i], kickstärke=kickstärke)

    return phi_array, p_array


def wenn_maus_geklickt(event, ax, kickstärke=2.6, anz_iterationen=1000):
    """Zeichne die Standardabbildung ausgehend von Mausposition."""
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        xpos = event.xdata
        ypos = event.ydata

        # Berechne Standardabbildung mit Mitte auf Maus, gegebener Kickstärke und Anzahl an Iterationen
        x, y = gesamt_iteration_standardabbildung(offset=np.array(
            [xpos, ypos]), kickstärke=kickstärke, anz_iterationen=anz_iterationen)

        # fuegt Kurve hinzu; marker ist festgelegt und ls ist leer, während die Farbe frei wechselt per default Einstellung
        # markersize ist klein gewählt für besseres erkennen der regulaeren Bereiche des Phasenraums
        ax.plot(x, y, marker='.', ls='', markersize=2)
        event.canvas.draw()                     # plotten


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)
    # Parameter
    kickstärke = 2.6
    anz_iterationen = 10000

    """Extra Aufgabe:

    iter=100
    phi_array_extra, p_array_extra=gesamt_iteration_standardabbildung(offset=np.array([1.1,0.6]),kickstärke=8.0,anz_iterationen=iter)
    ergebnis=[phi_array_extra[iter],p_array_extra[iter]] #ergebnis=5.244807229449704 -1.9866991190754115; betriebssystem= windwos10 +wsl ubuntu; cpu=Intel i5-7300U
    print(ergebnis)
    """

    # Erstelle einen Plotbereich
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, autoscale_on=False)
    ax.axis([0, 2.0*np.pi, -np.pi, np.pi])                  # Achsengrenzen
    # Beschriftung x-Achse
    ax.set_xlabel("Radialwinkel phi")
    # Beschriftung y-Achse
    ax.set_ylabel("Einheitenloser Impuls p/p_0")

    # definiert Subfunktion von "wenn_maus_geklickt"-Funktion, bei welcher ax, kickstärke und anz_iterationen festgelegt sind
    klick_funktion = functools.partial(
        wenn_maus_geklickt, ax=ax, kickstärke=kickstärke, anz_iterationen=anz_iterationen)

    # übergibt das mausklick-event an die "wenn_maus_geklickt"-funktion und zeichnet somit die Spirale an der geklickten Stelle
    fig.canvas.mpl_connect('button_press_event', klick_funktion)

    # Stellt den plot dar
    plt.show()


if __name__ == "__main__":
    main()

"""Als Aufgabe wurde gestellt den Phasenraum fur verschiedene Werte des Parameters K: K = 0; 0,2;
0,9; 2,1; 2,5; 6,0; 6,5 zu betrachten. Fuer K=K = 0; 0,2;
0,9; 2,1; 6,0; 6,5 waren nur chaotische Bereiche zu finden. Fuer K=2.5 waren reguläre Bereich in Form von Inseln zu sehen. 
Diese waren umgeben von Bereichen chaotischer Dynamik. Ich habe insgesamt vier Gruppen an solchen regulären Inseln entedeckt, 
wobei ich jede Fixpunktgruppe als jeweils eine Gruppe gezählt habe. 
Desweiteren betrachtete ich besonders den Bereich der Gruppe von p/p_0 von 0.14 bis 0.26 und phi von 2.14 bis 2.24.
Diese Gruppe liegt schon innerhalb einer Gruppe hat aber selbst in sich fraktalartig noch weiter Fixpunktgruppen.
Im Bereich p/p_0 von 0.290 bis 0.302 und phi von 2.135 bis 2.141 kann man den Grenzübergang von regulärer zu chaotischer Dynamik beobachten.
Hier war besonders interessant zu sehen, dass die im zwischen diesem Bereich und der anderen vorher erwähnten Gruppe zwar chaotisch sind, 
dennoch eine gewisse Ordnung nahe dem regulären Bereich haben.
Sie häufen sich entlang des Grenzbereichs und ebenfalls entlang einer Linie bei p/p_0= ca. 0.3."""
