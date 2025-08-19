"""Aufgabe 4: Differentialgleichungen


Das Programm  ermittelt die Lösung einer gegebenen Differentialgleichung und 
zeichnet die Trajektorie, sowie die stroboskopische Darstellung der Lösung in 
zwei Interaktive plots ein, nach vorgabe der Startbedingungen durch klicken 
innerhalb der plots.
"""

""" Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs
Code
die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die nächste
Zeile kommentiert/entfernt werden."""

# %matplotlib ipympl




import functools
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint          # Integrationsroutine fuer DGL
def abl(y, t, A=0.2, B=0.1, w=1):
    """Dies ist die rechte Seite der DGL, welche die jeweils erste Ableitung 
    des Ortes x und des Impulses p (welche in y gespeichert sind) zurückgibt. 
    A, B und w sind Parameter der DGL."""
    # x=y[0], p=y[1]
    return (y[1], -4 * y[0] ** 3 + 2 * y[0] - A - B * np.sin(w * t))


def Hamilton(x, p, t, A=0.2, B=0.1, w=1):
    """Dies ist die Hamiltonfunktion des Systems. x und p geben Ort und 
    Impuls an. A, B und w sind Parameter des Systems. t ist die Zeit zu 
    welcher die Hamiltonfunktion ausgewertet werden soll."""
    return p ** 2 / 2 + x ** 4 - x ** 2 + x * (
        A + B * np.sin(w * t))


def loesung_dgl(y0, A=0.2, B=0.1, w=1, anz_Datenpunkte=1000, t_ende=20):
    """Diese Methode löst die gegebene DGL und gibt die Trajektorie, sowie 
    die stroboskopische Darsetllung zurück. y0 ist der Startwert des Ort und 
    Impulses. A, B und w sind Parameter des Systems. anz_Datenpunkte gibt die 
    Anzahl an Datenpunkten welche für die Darstellung und Berechnung der 
    Trajektorie genutzt werden. t_ende gibt an bis zu welchem Zeitpunkt die 
    DGL ausgewertet wird."""
    # Schrittweite der stroboskopischen Zeiten
    step = (2 * np.pi / w)
    # stroboskopische Zeiten t_n
    zeiten_stroboskopisch = np.arange(0, t_ende,  step=step)
    anz_stroboskopisch = zeiten_stroboskopisch.size
    # da die stroboskopische Zeiten eine Teilmenge der Trajektorienzeiten
    # sein sollen, wird geprüft welche Anzahl größer ist.
    if anz_Datenpunkte > anz_stroboskopisch:
        # damit die Anzahl an  Trajektorienzeiten = anz_Datenpunkte wird hier
        # die Anzahl an stroboskopische Zeiten abgezogen
        anz_Datenpunkte -= anz_stroboskopisch
        # gleichverteilte Trajektorienzeiten
        zeiten_trajektorie = np.linspace(0, t_ende, num=anz_Datenpunkte)
        # Konkatenation der Trajektorienzeiten und stroboskopischen Zeiten
        zeiten = np.concatenate(
            (zeiten_stroboskopisch, zeiten_trajektorie))
    else:
        # falls die Anzahl an stroboskopischen Zeiten größer als die Anzahl
        # an gewünschten Trajektorienzeiten ist, wird die Anzahl auf
        # anz_Datenpunkte gekürzt
        zeiten = zeiten_stroboskopisch[:anz_Datenpunkte]
        anz_stroboskopisch = anz_Datenpunkte
    # y_t und odeint brauchen geordnete Zeiten, deshalb werden die Indexe
    # ermittelt, welche die Zeiten sortieren
    ordered_index = np.argsort(zeiten)
    # Integration der DGL
    y_t = odeint(abl, y0, zeiten[ordered_index], args=(A, B, w))
    # Um die Lösungen für die stroboskopische Zeiten wieder zu extrahieren
    # werden die Indexe ermittelt welche die Daten wieder in ihre
    # ursprüngliche Ordnung bringen.
    inverse_index = np.argsort(ordered_index)
    # Die ersten 'anz_stroboskopisch' Einträge des y_t arrays in
    # ursprünglicher Reihenfolge entsprechen den strobokopischen Punkten
    y_t_stroboskopisch = y_t[inverse_index][:anz_stroboskopisch]
    return y_t, y_t_stroboskopisch


def neuer_startpunkt(event, ax_dict, A=0.2, B=0.1, w=1, anz_Datenpunkte=1000,
                     t_ende=20):
    """Berechne und zeichne die Trajektorie und die stroboskopische 
    Darstellung ausgehend von Mausposition. event ist das 
    'button_press_event'. ax_dict ist das dictionary welches die subplots 
    speichert. A, B und w sind Parameter des Systems. anz_Datenpunkte gibt 
    die Anzahl an Datenpunkten welche für die Darstellung und Berechnung der 
    Trajektorie genutzt werden. t_ende gibt an bis zu welchem Zeitpunkt die 
    DGL ausgewertet wird."""

    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = event.canvas.toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        # Startwert ist der geklickte Punkt
        y0 = [event.xdata, event.ydata]
        # Lösung der DGL mittels loesung_dgl-Methode ermittelt
        y_t, y_t_stroboskopisch = loesung_dgl(
            y0, A, B, w, anz_Datenpunkte, t_ende)
        # Auslesen der Spalten mit x und p durch transponieren von y_t
        x_t, p_t = y_t.T
        x_t_stroboskopisch, p_t_stroboskopisch = y_t_stroboskopisch.T
        # plotten der Daten in gegebene plots
        ax_dict["stroboskopische Darstelung"].plot(
            x_t_stroboskopisch, p_t_stroboskopisch, marker=".", ms=1, ls="")
        ax_dict["Trajektorie"].plot(x_t, p_t, marker="", ms=1, ls="-")
        event.canvas.draw()                     # plotten
        return y_t, y_t_stroboskopisch


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)
    # Ausgabe der relevanten Parameter

    print(
        """Verwendete Parameter: 
        Die betrachtete Hamilton Funktion lautet: p ** 2 / 2 + x ** 4 - x ** 
        2 + x * (A + B * np.sin(w * t)). Die dafür verwendeten Parameter 
        lauten: A = 0.2, B = 0.1, w = 1. Die gewählte Anzahl der Datenpunkte 
        der Trajektorie lautet 5000. Die Zeit bis zu welcher die Trajektorie 
        und die Stroboskopische Betrachtung bestimmt werden sollte liegt bei 
        200 Perioden. Dies entspricht einer Zeit von 400 * pi. Der 
        betrachtete x Bereich wurde von -1.5 bis 1.4 gewählt. Der betrachtete 
        p Bereich wurde von -1.8 bis 1.8 gewählt. Die Energielevels der 
        Konturlinien wurde gleichverteilt im Bereich von -0.3 bis 1.2 gewählt.
        Für die Erstellung des Konturplots wurden 1000 * 1000 Punkte der 
        Hamilton-funktion berechnet.""")
    # Parameter

    # Parameter des Pendels
    A, B, w = 0.2, 0.1, 1
    # Anzahl der Datenpunkte der Trajektorie
    anz_Datenpunkte = 5000
    # Zeit bis zu welcher die Trajektorie und die Stroboskopische Betrachtung
    # bestimmt werden
    t_ende = 200 * 2 * np.pi
    # Achsengrenzen
    x_achsengrenze = [-1.5, 1.4]
    p_achsengrenze = [-1.8, 1.8]
    # Energielevels der Konturlinien
    levels = np.linspace(-0.3, 1.2, 8, endpoint=True)
    # Erstellung des Werte der Hamiltonfunktion für B=0 und t=0 für den
    # Konturplot
    x_contour = np.linspace(*x_achsengrenze, 1000)
    p_contour = np.linspace(*p_achsengrenze, 1000)
    x2d_contour, p2d_contour = np.meshgrid(x_contour, p_contour)
    Hamilton_xy_contour = Hamilton(
        x2d_contour, p2d_contour, A=A, t=0, B=0, w=w)
    # Hilfsliste an plots zur Zuordnung in dicts und ausgabe der plot-Titel
    plot_list = ["stroboskopische Darstelung", "Trajektorie"]
    # Dictionaries welche die 2 figures und subplots halten
    fig_dict = {}
    ax_dict = {}

    # Bei Mausklick soll die Funktion neuer_startpunkt aufgerufen werden,
    # wobei der Plotbereich ax und die Parameter `A`, `B` ,
    # `anz_Datenpunkte` , t_ende und `w` beim Aufruf
    # mit uebergeben werden:
    klick_funktion = functools.partial(
        neuer_startpunkt, ax_dict=ax_dict, A=A, B=B, w=w,
        anz_Datenpunkte=anz_Datenpunkte, t_ende=t_ende)

    # Erstellung der 2 Plots
    for plot in plot_list:
        fig = fig_dict[plot] = plt.figure()
        ax = ax_dict[plot] = fig.add_subplot(1, 1, 1)
        # Achsenbereiche setzen
        ax.set_xlim(*x_achsengrenze)
        ax.set_ylim(*p_achsengrenze)
        # Achsen labeln
        ax.set_xlabel("x_t")
        ax.set_ylabel("p_t")
        # Diagrammtitel
        ax.set_title("Differentialgleichung " + plot)
        # Erstellung und Labelung des Konturplots
        contour = ax_dict[plot].contour(
            x2d_contour, p2d_contour, Hamilton_xy_contour, levels=levels)
        ax.clabel(contour, contour.levels)

    for fig in fig_dict.values():
        # übergibt das mausklick-event an die "neuer_startpunkt"-funktion und
        # zeichnet somit die Phasenraumpunkte für den geklickten Startpunkt
        fig.canvas.mpl_connect(
            'button_press_event', klick_funktion)
    # stellt plots dar
    plt.show()
    # gibt Nutzerhilfe aus
    print("""Klicken Sie in einen der beiden plots um den Startpunkt für die 
          Lösung der Differentialgleichung vorzugeben. Nach dem Klicken wird 
          eine neue Trajektorie, sowie eine neue stroboskopische Darstellung 
          in die plots eingezeichnet.""")


if __name__ == "__main__":
    main()

    """a) Für B=0 sind im Phasendiagramm geschlossene Bahnen zu beobachten. 
    Die Bahn des Teilchens hat eine konstante Energie. Das Teilchen bewegt 
    sich entweder nur in einer von den beiden Mulden des Potentials oder 
    bewegt sich zwischen disen Mulden hin und her. Das Potential in welchem 
    sich das Teilchen befindet strebt für |x| -> inf ebenfalls gegen 
    unendlich. Auf Grund dessen gibt es nur geschlossene Teilchenbahnen.
    b) Für B=0.1 sind die Bahnen wesentlich verbreitert in ihrer Energie. 
    Startpositionen zwischen den Mulden mit geringem Impuls zeigen 
    chaotisches Verhalten. Diser chaotischer Bereich ist umgeben von 
    Bereichen regulärer Dynamik bei höheren |p|, welche der Dynamik für B=0 
    gleichen. In dem chaotischen Bereich sind reguläre Inseln für 
    Startpositionen nahe den Mulden und niedrigen Impulsen. Es gibt ebenfalls 
    noch weitere reguläre Bereiche z.B. die in c) erwähnte periodische 
    Trajektorie, sowie weitere periodische Trajektorien in dem chaotischen 
    Bereich. Diese periodischen Trajektorien können sich ebenfalls über einen 
    weiten Energiebereich erstrecken. Dabei wechseln sie periodisch zwischen 
    festen Energieleveln.
    c) Mittels zoomen im stroboskopischen Phasenraum habe ich eine 
    periodische Trajektorie mit m=4 gefunden. Die Startkoordiante dieser 
    liegt bei x=0.549 und p=-0.837. Die Periode t*= 8 * pi beträgt ungefähr 
    25.133 .
    """
