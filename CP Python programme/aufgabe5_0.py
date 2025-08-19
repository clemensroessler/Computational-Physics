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




from scipy.integrate import odeint          # Integrationsroutine fuer DGL
import matplotlib.pyplot as plt
import numpy as np
def ode_system(y, t, b, c, lockdown=False):
    """
    Diese Funktion gibt das Ergebnis der rechten Seite des 
    Differentialgleichungssystems (DGLS) zurück in Form eines Numpy-arrays.
    Die S, I und R Werte sind in y gespeichert nach dieser Vorschrift : 
    y[0]=S; y[1]=I; y[2]=R.
    t ist die Zeit zu welcher das DGLS ausgewertet wird. b und c sind 
    Parameter des DGLS. lockdown ist ein bool welcher bestimmt ob b Aufgrund 
    des Lockdowns reduziert werden sollte.
    """
    if lockdown == True:
        if t >= 20:
            if t >= 60:
                b = 0.5*b
            else:
                b = 0.3*b
    abl = np.zeros(3)
    abl[0] = -b * y[0] * y[1]
    abl[1] = (b * y[0] - c) * y[1]
    abl[2] = c * y[1]
    return abl


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)

    # Ausgabe der relevanten Parameter
    print("""Verwendete Parameter: Die Größe der simulierten Bevölkerung 
          lautet 80 Millionen. Die Parameter des SIR-Modells lauten: b = 0.5, 
          gamma = 0,1.Die Simulation startet mit einem Infizierten. Es werden 
          insgesamt Zeiten von t = 0 bis t = 150 betrachtet. Pro Zeitenheit 
          werden 100 Datenpunkte erfasst. Für die Auswertung der 
          Belastungsgrenze wurden 20 Datenpunkte genommen in einem Intervall 
          von b = 0,11 bis b = 0,5. Die Belastungsgrenze wurde bei 10% 
          angesetzt. Der simulierte Lockdown reduziert ab Tag t = 20 den b 
          Parameter auf 30% des Anfangswertes und erhöht ihn nach t = 60 auf 
          50%.""")

    # Ausgabe der Benutzerführung
    print("""Auf der linken Seite sehen sie die grafische Darstellung des 
          SIR-Modells. Das Modell wurde einmal mit und einmal ohne Lockdown 
          simuliert. In der Legende können sie sehen welche Kurve zu welcher 
          Simulation gehört. Auf der rechten Seite sehen sie den Plot des 
          relativen Maximums an Infizierter Bevölkerung in Abhängigkeit von 
          der Basisreproduktionszahl R_0.""")

    # Parameter
    # Bevölkerungszahl
    N = 8 * 10 ** 7
    # SIR Parameter
    b = 0.5
    c = 0.1
    # Startanzahl Infizierte
    I_start = 1.0
    # SIR Startwerte in Array Form
    y_start = np.array([N-I_start, I_start, 0])/N
    # Zeit Start- und Endwert
    t_start = 0
    t_ende = 150
    # Anzahl Datenpunkte pro Zeiteinheit
    t_step_anz = 100
    # Array an Zeitdatenpunkten
    zeiten = np.linspace(t_start, t_ende, t_ende*t_step_anz + 1)
    # Achsengrenzen
    t_achsengrenze = [t_start, t_ende]
    N_achsengrenze = [0.01, 1.05]

    # b-array fuer Belastungsgrenze
    b_anz = 20
    b_min = 0.11
    b_max = 0.5
    b_array = np.linspace(b_min, b_max, b_anz)
    R0_array = b_array / c
    R0_achsengrenze = [b_min / c, b_max / c]

    # DGL lösen
    # ohne Lockdown
    y_t = odeint(ode_system, y_start, zeiten, args=(b, c, False))
    # mit Lockdown
    y_t_lockdown = odeint(ode_system, y_start, zeiten, args=(b, c, True))
    # fuer verschiedene b-Werte (Belastungsgrenze)
    y_t_b = [odeint(ode_system, y_start, zeiten, args=(b, c))
             for b in b_array]

    # Erstellen der Plots
    fig = plt.figure(figsize=(14, 8))

    # SIR
    ax = fig.add_subplot(1, 2, 1)
    # Achsenbereiche setzen
    ax.set_xlim(*t_achsengrenze)
    ax.set_ylim(*N_achsengrenze)
    # Achsen labeln
    ax.set_xlabel("Zeit t")
    ax.set_ylabel("Anteil der Bevölkerung N_t/N")
    # Diagrammtitel
    ax.set_title("SIR-Modell")

    # Belastbarkeit
    ax2 = fig.add_subplot(1, 2, 2)
    # Achsenbereiche setzen
    ax2.set_xlim(*R0_achsengrenze)
    ax2.set_ylim(*N_achsengrenze)
    # Achsen labeln
    ax2.set_xlabel("Basisreproduktionszahl R_0")
    ax2.set_ylabel("Anteil der Bevölkerung N_t/N")
    # Diagrammtitel
    ax2.set_title("Maximale Auslastung des Gesundheitssystems")

    # Plot mit Farben und Beschriftungen fuer Legende fuer SIR
    # ohne Lockdown
    ax.plot(zeiten, y_t[:, 0], c='r', ls='-', label="S normal")
    ax.plot(zeiten, y_t[:, 1], c='g', ls='-', label="I normal")
    ax.plot(zeiten, y_t[:, 2], c='b', ls='-', label="R normal")
    # mit Lockdown
    ax.plot(zeiten, y_t_lockdown[:, 0], c='c', ls='-', label="S lockdown")
    ax.plot(zeiten, y_t_lockdown[:, 1], c='m', ls='-', label="I lockdown")
    ax.plot(zeiten, y_t_lockdown[:, 2], c='y', ls='-', label="R lockdown")
    # Legende
    ax.legend(loc='lower left')

    # Plot fuer Belastungsgrenze
    x = np.zeros(R0_array.size)
    for i in range(R0_array.size):
        I_max = np.max(y_t_b[i][:, 1])
        x[i] = I_max
    # Auslastungskurve
    ax2.plot(R0_array, x, ls='-', label="Absolute Maximale Auslastung")
    # Belastungsgrenze
    grenze = np.ones(R0_array.size) * 0.1
    ax2.plot(R0_array, grenze, ls='-', label="Belastungsgrenze", linewidth=3)
    # Legende
    ax2.legend(loc='upper right')

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
