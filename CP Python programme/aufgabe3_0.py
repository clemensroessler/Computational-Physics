"""Aufgabe 3: Elementare numerische Methoden II

Das Programm  ermittelt nach drei verschiedenen Methoden das Integrall 
einer Funktion in einem Intervall in Abhaengigkeit von h und vergleicht diese 
numerischen Methoden mit einer analytischer Auswertung.
"""

""" Das ipympl package ermöglicht das erstellen von interaktiven plots in Vs
Code
die solchen in Jupyter Notebook gleichen. Wenn nicht benötigt kann die nächste
Zeile kommentiert/entfernt werden."""

# %matplotlib ipympl




import matplotlib.pyplot as plt
import numpy as np
def sinh2(x):
    return np.sinh(2 * x)


def mittelpunkt_integration(fkt, a, b, N):
    """Integration durch Mittelpunkt-Regel:
    Die Funktion `fkt` wird auf dem Intervall [a, b]
    mit N Teilintervallen integriert.
    """
    x, h = np.linspace(start=a, stop=b, num=N, endpoint=False, retstep=True)
    return h * np.sum(fkt(x + h / 2))


def trapez_intergration(fkt, a, b, N):
    """Integration durch Trapez-Regel:
    Die Funktion `fkt` wird auf dem Intervall [a, b]
    mit N Teilintervallen integriert.
    """
    x, h = np.linspace(start=a, stop=b, num=N+1, endpoint=True, retstep=True)
    return h * (np.sum(fkt(x)) - (fkt(a) + fkt(b)) / 2)


def sympson_integration(fkt, a, b, N):
    """Integration durch Sympson-Regel:
    Die Funktion `fkt` wird auf dem Intervall [a, b]
    mit N Teilintervallen integriert.
    """
    x, h = np.linspace(start=a, stop=b, num=N, endpoint=False, retstep=True)
    return h / 6 * (2 * np.sum(fkt(x)) + 4 * np.sum(fkt(x + h / 2))
                    + fkt(b) - fkt(a))


def fehler_integration_analytisch(a, b, N, integration_name):
    """Gibt das analytisch erwartete Skalierungsverhalten des relativen 
    Fehler einer Integrationsmethode mit h zurück."""
    h = (b - a) / N
    fehler_dict = {"Mittelpunkt-Regel": 2,
                   "Trapez-Regel": 2,
                   "Sympson-Regel": 4}
    h_skalierung = fehler_dict[integration_name]
    return h ** h_skalierung


def relativer_fehler(numerisch, analytisch):
    """Vergleicht einen numerischen und einen analytischen Wert und gibt den 
    absoluten relativen Fehler der beiden zurück."""
    return np.abs(numerisch / analytisch - 1)


def main():
    """Hauptprogramm. Aufruf fuer verschiedene Parameter."""
    print(__doc__)
    # Ausgabe der relevanten Parameter
    print(
        """
    Die Untersuchte Funktion ist sinh(2 * x). Die Anzahl der gewählte 
    Datenpunkte beträgt 698 auf Grund des entfernens doppelter 
    Datenpunkte in einem Set von 1000 Datenpunkten. Die Grenzen der 
    Integration betragen a = -pi / 2 und b = pi / 4. Die Anzahl an 
    Intervallen der Integrationsmethoden liegt bei 1 bis 10 ** 5. Als 
    analytischer Ergebnis des Integralls wurde -4.541387398431732 angenommen.
    """
    )

    # Parameter
    anz_datenpunkte = 1000
    a = -np.pi / 2
    b = np.pi / 4
    funktion = sinh2
    N_start_power = 0
    N_end_power = 5
    integration_list = [mittelpunkt_integration,
                        trapez_intergration, sympson_integration]
    analytisch_dict = {"sinh2": -4.541387398431732}
    integrationsregel_dict = {
        "mittelpunkt_integration": "Mittelpunkt-Regel",
        "trapez_intergration": "Trapez-Regel",
        "sympson_integration": "Sympson-Regel"}

    # Deklariert einen numpy-Array mit logaritmisch Verteilten Werten an
    # Intervallanzahlen als Integer
    N_array = np.logspace(N_start_power, N_end_power,
                          num=anz_datenpunkte, dtype=int)
    # Durch das runden auf Integerwerte gibt es einige Doppelungen in N_array,
    # welche durch die folgende Zeile entfernt werden.
    N_array = np.unique(N_array)
    h_array = (b - a) / N_array
    analytisch_ergebnis = analytisch_dict[funktion.__name__]

    # initialisiert dictioniaries in welchen die berechneten Arrays
    # gespeichert werden für das spätere plotten dieser Arrays
    fehler_analytisch_dict = {}
    fehler_numerisch_dict = {}

    # loopt über alle gegebenen Integrationsmethoden und berechnet die
    # relativen Fehler und speichert diese in dicts ab
    for integrations_methode in integration_list:
        # integrations_bezeichnung sind die keys für die dicts, welche später
        # genutzt wird um Schreibarbeit beim labelen des plots zu sparen
        integrations_bezeichnung = integrationsregel_dict[
            integrations_methode.__name__]
        # berechnet das numerische Ergebnis für alle N aus N_array
        numerisch_ergebnis = np.array([integrations_methode(
            funktion, a, b, N) for N in N_array])
        # berechnet den analytischen Fehler für alle N aus N_array
        fehler_analytisch_dict[
            integrations_bezeichnung] = [fehler_integration_analytisch(
                a, b, N, integrations_bezeichnung) for N in N_array]
        # vergleicht das numerische und das analytische Ergebnis und bildet
        # daraus den numerischen Fehler
        fehler_numerisch_dict[integrations_bezeichnung] = relativer_fehler(
            numerisch_ergebnis, analytisch_ergebnis)

    # Erstelle einen Plotbereich
    fig = plt.figure()
    # sorgt für doppelt logaritmische Axendarstellung
    ax = fig.add_subplot(1, 1, 1, xscale="log", yscale="log")
    # Achsengrenzen
    ax.axis([h_array[0], h_array[h_array.size - 1], 10 ** (-16), 1])
    # Beschriftung x-Achse
    ax.set_xlabel("Intervallbreite h")
    # Beschriftung y-Achse
    ax.set_ylabel("Betrag des relativen Fehlers")

    for integrations_bezeichnung in fehler_analytisch_dict:
        # plottet numerische Datenpunkte
        ax.plot(
            h_array, fehler_numerisch_dict[integrations_bezeichnung], ls='',
            marker='o', mew=0, ms=3, label=integrations_bezeichnung)
        # plottet analytische Datenpunkte
        ax.plot(h_array, fehler_analytisch_dict[integrations_bezeichnung],
                ls='', marker='o', mew=0, ms=3, label=(
            integrations_bezeichnung + ' analytisch'))

    # erstellt Legende
    ax.legend()
    # erstellt dynamischen Plottitel
    ax.set_title('Relativer Fehler der Integrationsmethoden')
    # stellt einen plot für jede Integrationsmethode dar
    plt.show()


if __name__ == "__main__":
    main()

    """
    Analytische Ergebnisse:
        a)  -4.541387398431732
        b)  0.1772453850905516
        c)  0.7853981633974483   
    
    Fehlerverhalten:
    a) Die relativen Fehler verlaufen alle proportional zu der analytisch 
    vorhergesagten h-Skalierung. Das Anwenden der Mittelpunkt-Regel führt zu  
    einem kleineren Fehler als bei der Trapez-Regel um ca. einen Faktor 2.
    Der Fehler fällt wie erwartet deutlich schneller für die Sympson-Regel, 
    sodass bei ca. h = 10 ** -3 der minimale relative Fehler erreicht wird.
    Danach verläuft der relative Fehler für die Sympson-Regel ca. konstant, 
    da der Rundungsfehler hier dem Diskretisierungsfehler überwiegt.
    b) Beim Anwenden der Integrationsmethoden auf die e-funktion kann man 
    beobachten wie sich das Verhalten der Intervallbreite und dem relativen 
    Fehler zwischen den Methoden gleicht. 
    Bei einem h-Wert von 0.5 und größer sind die relativen Fehler noch von 
    der selben Größenordnung wie das analytisch Vorhergesagte Verhalten. Für 
    h kleiner als 1/10 ist der numerische relative Fehler aller 3 Methoden 
    kleiner als das analytisch Vorhergesagte Skalierungsverhaten für die 
    Sysmpson-Regel und der Fehler fällt rapide ab. Für h= 6 * 10 ** -2 und 
    kleiner gehen die Fehler in das konstante Minimum bei Fehler = 10 ** -15 
    über. Dieses Verhalten ist, über das approximative Verschwinden der 
    Funktion an den Rändern zu erklären. Die e-funktion, sowie die 
    Ableitungen dieser sind an den Rändern des Integrationsbereichs 
    verschwindend klein. Da der Fehler nicht nur mit h, sondern ebenfalls mit 
    dem Funktionswert der ersten, bzw. dritten Ableitung skaliert, erklärt 
    dies den Verlauf der Fehlerkuren.
    c) Die Fehlerkurven verhalten sich für die heaviside-funktion ebenfalls 
    gleich unabhängig von der Integrationsmethode. Der relative Fehler sinkt 
    für alle drei Methoden bei wachsender Intervallbreite, jedoch langsamer 
    als analytisch erwartet. Dies ist durch den Fehler am unstetigen Nullunkt 
    der heaviside-funktion zu erklären. Die konstaten Bereiche der Funktion 
    tragen nicht zum Fehler bei. Sollte ein Intervall jedoch den Nullpunk 
    überschreiten so gibt es hier unabhängig von der größe des Intervalls 
    einen von 0 verschiedenen Fehler. Bei kleineren Intervallen hat dieser 
    Fehler jedoch weniger Gewicht im Endergebnis, weshalb der relative Fehler 
    trotzdem mit fallendem h sinkt. Bei der Mittelpunkt- und bei der 
    Trapez-Regel sind Punkte zu beobachten welche scheinbar zufällig 
    innerhalb des h Intervalls liegen und welche einen Fehler der 
    Größenordnung 10 ** -15 aufweisen. Dieses Phänomen tritt auf, da zufällig 
    für diese Intervallbreiten eine Intervallgrenze sehr na beim Nullpunkt 
    gesetzt wurde und somit ein Diskretisierungsfehler kleiner als 10 ** -15 
    erzeugt wird.
    """
