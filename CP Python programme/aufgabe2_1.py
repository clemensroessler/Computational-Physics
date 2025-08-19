"""Aufgabe 2: Elementare numerische Methoden I

Das Programm  ermittelt nach drei verschiedenen Methoden die Ableitung 
einer Funktion an einer Stelle in Abhaengigkeit von h und vergleicht diese 
numerischen Methoden mit einer analytischer Auswertung.
"""

""" Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs
Code
die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die nächste
Zeile kommentiert/entfernt werden."""

# %matplotlib ipympl




import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
def atan_funktion(x):
    """ Dies ist die Funktion an welchem Beispiel die numerische Ableitungen 
    untersucht werden"""
    return np.arctan(x ** 4)


def atan_ableitung(x):
    """ Gibt den Wert der analytischen Ableitung der untersuchten Funktion
    zurück"""
    return (4 * x ** 3) / (x ** 8 + 1)


def vorwaertsdifferenz(fkt, x_0=0, h=10 ** (-6)):
    """ Die erste Numerische Ableitungsmethode; gibt Ableitung an Stelle x_0 
    der Funktion fkt zurück, mit variabler Schrittweite h """
    return (fkt(x_0 + h) - fkt(x_0)) / h


def zentraldifferenz(fkt, x_0=0, h=10 ** (-6)):
    """ Die zweite Numerische Ableitungsmethode; gibt Ableitung an Stelle x_0 
    der Funktion fkt zurück, mit variabler Schrittweite h """
    return (fkt(x_0 + (h / 2)) - fkt(x_0 - (h / 2))) / h


def extrapolierte_differenz(fkt, x_0=0, h=10 ** (-3)):
    """ Die dritte Numerische Ableitungsmethode; gibt Ableitung an Stelle x_0 
    der Funktion fkt zurück, mit variabler Schrittweite h """
    return (8 * (fkt(x_0 + (h / 4)) - fkt(x_0 - (h / 4))) -
            (fkt(x_0 + (h / 2)) - fkt(x_0 - (h / 2)))) / (3 * h)


def relativer_fehler_ableitung(fkt, ableitung_numerisch,
                               ableitung_analytisch, x_0=0, h=10 ** (-6)):
    """gibt den relativen Fehler der numerischen Ableitungsmethode zurück im 
    Vergleich zur analytischen Methode. Die Methoden werden an der Stelle x_0 
    und bei Schrittweite h verglichen."""
    return np.abs((ableitung_numerisch(fkt, x_0=x_0, h=h) /
                   ableitung_analytisch(x_0)) - 1)


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)

    # Parameter
    data_points = 1000
    x_0_array = np.array([1/4, -1/5])
    h_start = 10 ** (-10)
    h_end = 1

    # erstellt ein dictionary welches jeder Ableitfunktion einen
    # beschreibenden String als key zuordnet

    ableitungen_dict = {'Vorwärtsdifferenz': vorwaertsdifferenz,
                        'Zentraldifferenz': zentraldifferenz,
                        'extrapolierte Differenz': extrapolierte_differenz}

    # h_array ist ein numpy array logaritmisch angeordneter punkte von 10 **
    # -10 bis 1
    h_array = np.power(10, np.linspace(np.log10(h_start),
                       np.log10(h_end), data_points, endpoint=True))

    # erstellt zwei sets an plot für x_0=1/4 und x_0=-1/5
    for x_0 in x_0_array:

        # Erstelle einen Plotbereich
        fig = plt.figure()
        # sorgt für doppelt logaritmische Axendarstellung
        ax = fig.add_subplot(1, 1, 1, xscale="log", yscale="log")
        # Achsengrenzen
        ax.axis([h_start, h_end, 10 ** (-15), 1])
        # Beschriftung x-Achse
        ax.set_xlabel("Schrittweite h")
        # Beschriftung y-Achse
        ax.set_ylabel("Betrag des relativen Fehlers")

        # erstellt dictionary des analytisch erwarteten skalierungsverhalten
        # des relativen Fehler mit h
        analytischer_fehler_dict = {}
        # berechnet das analytisch erwartete Verhalten des Fehler
        x = sp.symbols('x')
        funktion = sp.atan(x ** 4)
        analytischer_fehler_dict['Vorwärtsdifferenz'] = np.abs((
            sp.diff(funktion, 'x', 2) /
            sp.diff(funktion, 'x')).subs('x', x_0) / 2) * h_array
        analytischer_fehler_dict['Zentraldifferenz'] = np.abs((
            sp.diff(funktion, 'x', 3) /
            sp.diff(funktion, 'x')
        ).subs('x', x_0) / 24) * np.power(h_array, 2)
        analytischer_fehler_dict['extrapolierte Differenz'] = np.abs((
            sp.diff(funktion, 'x', 5) /
            sp.diff(funktion, 'x')
        ).subs('x', x_0) / -64) * np.power(h_array, 4)
        # Wiederholt Berechnung und Darstellung für alle gegebenen
        # Ableitungsmethoden
        for ableitung_name in ableitungen_dict.keys():
            # berechnet einen array des relativen Fehler für die gegebenen
            # h-Werte in h_array
            relativer_fehler_array = relativer_fehler_ableitung(
                atan_funktion, ableitungen_dict[ableitung_name],
                atan_ableitung,
                x_0=x_0, h=h_array)

            # plottet numerische Datenpunkte
            ax.plot(h_array, relativer_fehler_array,
                    ls='-', label=ableitung_name)

            # plottet analytische Datenpunkte
            ax.plot(h_array, analytischer_fehler_dict[ableitung_name],
                    ls='-', label=(ableitung_name + ' analytisch'))
            # erstellt Legende
            ax.legend()
            # erstellt dynamischen Plottitel
            ax.set_title(
                'Relativer Fehler der Ableitungen bei x_0 = ' + str(x_0))
        # stellt einen plot für jedes x_0 dar
        plt.show()


if __name__ == "__main__":
    main()


"""
a) Bei kleinen h wird für die Vorwärtsdifferent eine sehr kleine Differenz 
durch ein sehr kleines h geteilt. Dadurch werden Rundungsfehler relevanter 
als Diskretisierungsfehler und der relative Fehler steigt bei fallendem h. 
Bei besseren Algorithmen mit kleineren Diskretisierungsfehlern tritt dieser 
Effekt schon bei größeren h auf.

b) Die numerischen Fehlerkurven lassen sich in zwei Bereiche einteilen: einen 
Bereich der durch Rauschen dominiert wird und ein glatter Bereich.
Da der Rauschbereich abhängig von Dingen wie CPU oder Betriebssystem ist, 
sind Aussagen über das reproduzierbare Verhalten nur im glatten Bereich 
sinnvoll. Für die Wahl des optimalen h wurde also ein Punkt im glatten 
Bereich ausgewählt, welcher den relativen Fehler minimiert. Diese Auswahl 
geschah per Augenmaß und lautet wie folgt:

                    h               Fehler
Vorwärtsdifferenz:  1.3 * 10 ** -8  1.0 * 10 ** -7 
Zentraldifferenz:   6 * 10 ** -6    2.0 * 10 ** -10
extr. Differenz:    5 * 10 ** -3    1.0 * 10 ** -12
"""
