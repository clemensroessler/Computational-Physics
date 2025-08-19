"""Aufgabe 0: Spirale

Beim Klicken innerhalb des Plotfensters wird eine Spirale
ab der Position des Mauszeigers gezeichnet.
"""

""" Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs Code 
die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die nächste Zeile kommentiert/entfernt werden."""

"""Bei Benutzung von Spyder war es notwendig für den Interaktiven Plot einzustellen. Dies geht wie folgt: Tools>prefrences>Ipthyon console>Graphics>Graphics backend>Backend: Automatic"""

import matplotlib.pyplot as plt
import numpy as np
import functools
import math
#%matplotlib ipympl


def Polarradius(t):
    """gibt den Radius r einer Spirale zurück in Abhängigkeit vom Parameter t"""
    return (1/2)**t


def Polarwinkel(t):
    """gibt den Polarwinkel phi einer Spirale zurück in Abhängigkeit vom Parameter t"""
    return 2*t*math.pi


def PolarZuKartesisch(r, phi):
    """Konvertiert gegebene Polarkoordinaten zu Kartesischen Koordinaten und gibt diese zurück"""
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y


def SpiralenErzeugende(Mitte=np.zeros(2), Windungen=5, Datenpunkte=1000):
    """Erzeugt zwei Numpy Arrays mit entweder kartesischen x- oder y-Koordinaten einer Spirale mit variabler Anzahl an Windungen, Mittelpunkt, und Anzahl an Datenpunkten"""
    t = np.linspace(
        0.0, Windungen, Datenpunkte)  # numpy array des parameter t für gegebene Anzahhl an Windungen und Datenpunkten
    Radius = Polarradius(t)  # numpy array der Radien r für alle t
    Winkel = Polarwinkel(t)  # numpy array der Winkel phi für alle t
    # Umwandlung des Radius und des Winkels in x und y numpy arrays
    xArray, yArray = PolarZuKartesisch(Radius, Winkel)
    xArray += Mitte[0]  # verschiebt die Spirale in die Mitte
    yArray += Mitte[1]  # verschiebt die Spirale in die Mitte
    return xArray, yArray


def wenn_maus_geklickt(event, ax, Windungen=5, Datenpunkte=1000):
    """Zeichne Spirale ausgehend von Mausposition."""
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        xpos = event.xdata
        ypos = event.ydata

        # Berechne Spirale mit Mitte auf Maus und gegebener Anzahl an Windungen und Datenpunkten
        x, y = SpiralenErzeugende(Mitte=np.array(
            [xpos, ypos]), Windungen=Windungen, Datenpunkte=Datenpunkte)

        ax.plot(x, y, ls='-', lw=1, c='r')      # Kurve hinzufuegen
        event.canvas.draw()                     # plotten


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)
    # Parameter
    Windungen = 5
    Datenpunkte = 1000

    # Erstelle einen Plotbereich
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, autoscale_on=False, aspect=1.0)
    ax.axis([-2.0, 2.0, -2.0, 2.0])                  # Achsengrenzen
    ax.set_xlabel("x")                                  # Beschriftung x-Achse
    ax.set_ylabel("y")                                  # Beschriftung y-Achse

    # erzeugt eine Spirale in der Mitte
    x, y = SpiralenErzeugende(Windungen=Windungen, Datenpunkte=Datenpunkte)
    ax.plot(x, y, ls='-', lw=1, c='b')

    # definiert Subfunktion von "wenn_maus_geklickt"-Funktion, bei welcher ax, Windungen und Datenpunkte festgelegt sind
    klick_funktion = functools.partial(
        wenn_maus_geklickt, ax=ax, Windungen=Windungen, Datenpunkte=Datenpunkte)

    # übergibt das mausklick-event an die "wenn_maus_geklickt"-funktion und zeichnet somit die Spirale an der geklickten Stelle
    fig.canvas.mpl_connect('button_press_event', klick_funktion)

    # Stellt den plot dar
    plt.show()


if __name__ == "__main__":
    main()
    
