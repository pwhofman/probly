# Wir holen uns matplotlib.pyplot und nennen es nur kurz "plt".
# Damit machen wir am Ende den Plot (Figur anzeigen, Titel setzen, usw.).
import matplotlib.pyplot as plt

# Numpy brauchen wir, um mit Arrays/Vektoren zu arbeiten
# (z.B. für die MLE-Werte und für die Winkel auf dem Kreis).
import numpy as np

# Ab hier kommen Sachen direkt aus matplotlib, die wir brauchen,
# um dann die eigene "Radar"-Projektion zu bauen.
# Das ist wichtig, damit matplotlib weiß, wie unser Spider-Plot gezeichnet wird.

# Circle und RegularPolygon sind Formen 
# die wir als Rahmen (Frame) für den Plot benutzen.
from matplotlib.patches import Circle, RegularPolygon

# Path repräsentiert Pfade/Linienverläufe (z.B. für den Polygon-Rand).
from matplotlib.path import Path

# register_projection benutzen wir, um matplotlib zu sagen:
# "Hey, es gibt jetzt eine neue Projektion namens 'radar'."
from matplotlib.projections import register_projection

# PolarAxes ist die Basis-Klasse für Polarkoordinaten (also Kreisdiagramme).
# Wir erben später davon, um unser eigenes RadarAxes zu bauen.
from matplotlib.projections.polar import PolarAxes

# Spine ist der Rahmen/Umrandung der Achsen (z.B. der äußere Rand des Plots).
from matplotlib.spines import Spine

# Affine2D brauchen wir, um unseren Polygon-Rahmen zu skalieren und zu verschieben,
# damit er genau in den Plot reinpasst.
from matplotlib.transforms import Affine2D


# ----------------------------------------------------------
# Funktion, die unsere Radar-/Spider-Struktur vorbereitet
# ----------------------------------------------------------
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.
    num_vars = Anzahl der Achsen (also z.B. Anzahl der Klassen).
    frame = Form des Rahmens: 'circle' oder 'polygon'.
    """

    # Wir berechnen gleichmäßig verteilte Winkel auf dem Kreis.
    # 0 bis 2*pi, ohne den Endpunkt (endpoint=False),
    # damit sich Anfang und Ende nicht doppeln.
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    # -----------------------------------------
    # Eigene Transform-Klasse für Radar
    # -----------------------------------------
    class RadarTransform(PolarAxes.PolarTransform):
        # Diese Methode sorgt dafür, dass unsere Linien
        # wirklich als Gerade zwischen Punkten gezeichnet werden
        # (statt als runde Kreisbögen, was davor das Problem war).
        def transform_path_non_affine(self, path):
            # Wenn der Pfad mehr als einen Interpolationsschritt hat,
            # dann interpolieren wir ihn auf num_vars Punkte.
            # So wird z.B. das Gitter in Polygonform gezwungen.
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            # Danach wird der Pfad mit der normalen Polar-Transformation umgerechnet.
            return Path(self.transform(path.vertices), path.codes)

    # -----------------------------------------
    # Eigene Achsen-Klasse für Radarplots
    # -----------------------------------------
    class RadarAxes(PolarAxes):
        # Name der Projektion – damit können wir später sagen:
        # subplot_kw=dict(projection='radar')
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            # Wir rufen den Konstruktor von PolarAxes auf (Standardverhalten),
            # und erweitern es dann.
            super().__init__(*args, **kwargs)
            # Wir drehen den Plot so, dass die erste Achse oben (Norden) ist.
            # 'N' = North.
            self.set_theta_zero_location('N')

        # fill() überschreiben:
        # Standardmäßig macht matplotlib die Fläche nicht zu.
        # Wir wollen aber geschlossene Flächen, daher closed=True als Standard.
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        # plot() überschreiben:
        # Auch hier sorgen wir dafür, dass Linien automatisch geschlossen werden
        # (Startpunkt wird am Ende wieder angehängt).
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        # Hilfsfunktion, um eine Linie zu "schließen".
        def _close_line(self, line):
            # x = Winkel, y = Radius
            x, y = line.get_data()
            # Wenn der erste Punkt nicht der letzte ist,
            # hängen wir den ersten Punkt nochmal ans Ende an.
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        # Labels für die Achsen (also die Klassen-Namen) setzen.
        def set_varlabels(self, labels):
            # set_thetagrids erwartet Winkel in Grad, also wandeln wir um.
            self.set_thetagrids(np.degrees(theta), labels)

        # Diese Methode erzeugt den Hintergrund-Patch (also die Fläche,
        # auf der unser Plot gezeichnet wird).
        def _gen_axes_patch(self):
            if frame == 'circle':
                # Kreis als Rahmen: Mittelpunkt (0.5, 0.5), Radius 0.5 (alles in Achsen-Koordinaten).
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                # Polygon als Rahmen: num_vars = Anzahl der Ecken.
                # radius=.5 heißt wieder: passt genau ins Quadrat der Achsen.
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                # Falls jemand etwas anderes als 'circle' oder 'polygon' übergibt,
                # werfen wir einen Fehler.
                raise ValueError(f"Unknown value for 'frame': {frame}")

        # Diese Methode erzeugt die "Spines" – also den äußeren Rand,
        # der den Plot umgibt.
        def _gen_axes_spines(self):
            if frame == 'circle':
                # Beim Kreis können wir einfach das Standardverhalten übernehmen.
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # Hier bauen wir einen Polygon-Rand.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon macht ein Vieleck mit Radius 1 um (0,0).
                # Wir skalieren es auf Radius 0.5 und verschieben es nach (0.5, 0.5),
                # damit es genau in unseren Achsenrahmen passt.
                spine.set_transform(
                    Affine2D().scale(.5).translate(.5, .5) + self.transAxes
                )
                # Wir geben ein Dictionary zurück: Schlüssel 'polar'
                # wird vom Rest von matplotlib erwartet.
                return {'polar': spine}
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    # Hier registrieren wir unsere neue Projektion-Klasse bei matplotlib.
    # Ab jetzt kann man "projection='radar'" in subplots verwenden.
    register_projection(RadarAxes)

    # Wir geben die Winkel zurück, damit wir sie später für die Daten nutzen können.
    return theta


# ===================== HIER BEGINNT DEIN CREDAL-PLOT =====================

# CIFAR-10 Klassen wie im Paper.
# Jede dieser Klassen wird später eine Achse im Spider-Plot.
labels = ["airplane", "truck", "ship", "horse", "frog",
          "dog", "deer", "cat", "bird", "automobile"]

# N ist einfach die Anzahl der Klassen (also 10).
N = len(labels)

# Spider-Winkel berechnen:
# Wir rufen unsere radar_factory-Funktion auf.
# frame='polygon' bedeutet: Rahmen / Gitter wird als Vieleck gezeichnet.
theta = radar_factory(N, frame='polygon')

# --- UNSERE SPÄTEREN MLE-WERTE ---
# Das ist ein NumPy-Array mit Wahrscheinlichkeiten für jede Klasse.
# Das sind nur Beispiel-Werte. Später benutzen wir wahrscheinlich andere Werte
mle = np.array([0.05, 0.03, 0.02, 0.04, 0.05,
                0.70, 0.03, 0.02, 0.04, 0.02])


# Wir erstellen eine neue Figur und ein Achsen-Objekt.
# figsize=(6, 6) bestimmt die Größe des Fensters.
# subplot_kw=dict(projection='radar') sagt:
# "Benutze unsere Radar-Projektion, die wir oben registriert haben."
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))

# Spinnennetz-Grid:
# set_rgrids legt die Kreise im Inneren fest (0.2, 0.4, 0.6, 0.8, 1.0).
# Das sind die Radius-Werte, die im Gitter angezeigt werden.
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])

# set_ylim gibt den minimalen und maximalen Radius an.
# 0.0 = Mittelpunkt, 1.0 = äußerer Rand.
ax.set_ylim(0.0, 1.0)

# Klassenlabels außen auf die Achsen schreiben.
# Unsere RadarAxes-Klasse hat dafür die Methode set_varlabels.
ax.set_varlabels(labels)

# ---------- MLE: roter Punkt ----------

# np.argmax(mle) gibt den Index der größten Wahrscheinlichkeit zurück.
# Also die Klasse, die das Modell am wahrscheinlichsten findet.
idx_mle = np.argmax(mle)

# angle_mle ist der Winkel der MLE-Klasse (also welche Achse).
angle_mle = theta[idx_mle]

# radius_mle ist der dazugehörige Wahrscheinlichkeitswert.
radius_mle = mle[idx_mle]

# ax.scatter zeichnet einen Punkt.
# [angle_mle] und [radius_mle] sind Listen, weil scatter mehrere Punkte
# auf einmal bekommen könnte – wir geben aber nur einen.
# s=80 -> Punktgröße.
# color='red' -> Punkt wird rot.
# label='MLE' -> Text für die Legende.
ax.scatter([angle_mle], [radius_mle], s=80, color='red', label='MLE')

# Titel oben über den Plot schreiben.
# pad=20 heißt: ein bisschen Abstand zum Plot.
plt.title("Credal Prediction", pad=20)

# Legende anzeigen (die benutzt das 'label', das wir bei scatter angegeben haben).
# loc='upper right' -> oben rechts,
# bbox_to_anchor verschiebt die Legende leicht nach außen,
# damit sie nicht auf dem Plot liegt.
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

# tight_layout sorgt dafür, dass nichts abgeschnitten wird
# (z.B. Labels am Rand).
plt.tight_layout()

# Und am Ende: Plot-Fenster anzeigen.
plt.show()