# PlanDigitalizer

Online-Werkzeug zur **Digitalisierung alter Baupläne und Handzeichnungen**.

**Website öffnen → Plan hochladen → automatische Vektorisierung → PDF / DXF / IFC / 3D herunterladen → in CAD oder BIM weiterbearbeiten.**

Die gesamte Verarbeitung läuft **im Browser**. Pläne werden nicht hochgeladen, nicht gespeichert und an niemanden übertragen – auch nicht an den Webserver. Die Website braucht deshalb keinen Server, keine Datenbank und keinen Login.

---

## Inhalt

1. [Was das Werkzeug kann](#was-das-werkzeug-kann)
2. [Erkennung: nur das Wesentliche](#erkennung-nur-das-wesentliche)
3. [Online stellen (Cloudflare Pages)](#online-stellen-cloudflare-pages)
4. [Datenschutz](#datenschutz)
5. [IFC und 3D-Modell](#ifc-und-3d-modell)
6. [Technik](#technik)
7. [Grenzen und nächste Schritte](#grenzen-und-nächste-schritte)
8. [Für Entwickler: lokal testen, Tests, Modell trainieren](#für-entwickler)
9. [Lizenzen und Quellen](#lizenzen-und-quellen)

---

## Was das Werkzeug kann

| Bereich | Umfang |
|---|---|
| **Eingabe** | JPG, PNG, WEBP, TIFF, PDF (erste Seite) – Scan, Handyfoto, Handskizze |
| **Ausrichtung** | Schräglage bis ±45° aus den erkannten Wänden · Perspektive entzerrt (Blatt auf Tisch *oder* schräg fotografiert ohne sichtbaren Blattrand) · Blaupausen/Negative |
| **Bildkorrektur** | Schatten/Hintergrund, Kontrast, Rauschen, Schwarz-Weiss, Lückenschluss |
| **Erkennung** | Wände, Fenster, Türen pro Pixel durch ein lokal laufendes neuronales Netz; Möbel, Massketten, Schraffuren, Texte werden als „nicht wesentlich“ erkannt |
| **Geometrie** | Wände als exakte, achsparallele Umrisse (Kanten auf die gezeichnete Linie eingerastet), Ecken und T-Stösse geschlossen; Öffnungen sauber eingeschnitten; Fenster und Türen als genormte Symbole (Anschlagseite aus dem gezeichneten Bogen) |
| **Optional** | „Auch Möblierung, Bemassung und übrige Linien übernehmen“ – dann zusätzlich alle restlichen Linien, Bögen, Kreise, Bemassung |
| **Texterkennung** | Deutsch + Englisch, auch senkrechte Masszahlen → editierbarer CAD-Text |
| **Massstab** | Kalibrierung (zwei Punkte + bekannte Länge) · automatisch aus Massketten · „M 1:100" + Papiergrösse |
| **Ausgabe** | Vektor-PDF (ein-/ausblendbare Ebenen) · DXF (Layer, mm) · IFC4 · 3D als GLB und OBJ |

**CAD-Layer:** `WALLS` (Wandumrisse), `HATCH` (Wandfüllung), `DOORS`, `WINDOWS`, `STAIRS`, `TEXT` (Raumnamen) – mit der Option zusätzlich `DIMENSIONS`, `LINES`, `SYMBOLS`

**DWG:** Echtes DWG lässt sich im Browser nicht erzeugen (das Format ist proprietär; der kostenlose ODA-Konverter ist ein Desktop-Programm). Die DXF öffnet sich in jedem CAD und lässt sich dort als DWG speichern.

**Dauer:** Erster Besuch ca. 62 MB Download (danach aus dem Browser-Zwischenspeicher, Start in wenigen Sekunden). Ein Plan braucht je nach Grösse und Rechner 15–40 Sekunden.

---

## Erkennung: nur das Wesentliche

Bestandespläne enthalten viel, das für die Weiterbearbeitung stört: Möblierung, Sanitärapparate, Massketten, Schraffuren, Beschriftungen, Plankopf. PlanDigitalizer übernimmt standardmässig nur, was den Bau beschreibt:

| Übernommen | Verworfen |
|---|---|
| Wände (Lage, Stärke, Ecken, T-Stösse) | Möbel, Küche, Sanitärapparate |
| Fenster und Türen (Breite, Lage, Anschlag) | Massketten, Masszahlen, Höhenkoten |
| Treppen | Boden- und Wandschraffuren |
| Raumnamen und Flächenangaben | Achsraster, Nordpfeil, Planrahmen, Plankopf |

**So funktioniert es**

1. **Pixel-Erkennung:** Ein kleines neuronales Netz (U-Net, 0.6 Mio. Parameter, 2.3 MB) ordnet jedem Bildpunkt *Wand*, *Fenster*, *Tür* oder *anderes* zu. Es läuft lokal im Browser über OpenCV – keine Cloud. Die Rechengrösse passt sich automatisch der Wandstärke im Bild an.
2. **Ausrichtung:** Aus den Kanten der erkannten Wände (nicht aus Möbeln oder Text) werden Schräglage (bis ±45°) und – bei schräg fotografierten Plänen – die Fluchtpunkte bestimmt. Der Plan wird entzerrt, gerade gedreht und neu erkannt.
3. **Rekonstruktion:** Die Wandfläche wird in achsparallele Wandabschnitte zerlegt; jede Wandkante wird auf die tatsächlich gezeichnete Linie eingerastet. Öffnungen werden auf die Wandstärke ausgerichtet und sauber ausgeschnitten.
4. **Symbole:** Fenster erhalten Leibungs- und Glaslinien, Türen Blatt und Anschlagbogen. Drehpunkt und Aufschlagseite werden aus dem im Plan gezeichneten Bogen gelesen; Doppeltüren werden erkannt.
5. **Massstab:** aus Kalibrierung, Massketten oder „M 1:100“ + Papiergrösse. Fehlt alles, wird er aus den Türbreiten **geschätzt** (deutlich als Schätzung markiert).

**Training.** Das Netz wurde mit 7'000 Plänen trainiert, die aus den Geometrien von 17'000 echten Wohnungsgrundrissen (Datensatz *ResPlan*, CC BY 4.0) erzeugt wurden – jeweils in zufälligem Zeichenstil nach Schweizer Gepflogenheiten: Wände schwarz, grau angelegt, als Umriss, mit Mauerwerk- oder Kreuzschraffur, als Handskizze; Türen mit vollem oder gestricheltem Bogen, Doppel- und Schiebetüren; Fenster mit Rahmen, Glas und Fensterbank; dazu Möblierung, Massketten mit Hochzahlen, Raumbeschriftungen, Treppen, Bodenbeläge, Achsraster, Plankopf sowie Alterung (Vergilbung, Bleistift, Flecken, Unschärfe, JPEG, Verzug, Perspektive). Geprüft wird mit Plänen, die das Netz nie gesehen hat, und mit echten Zeichnungen (ROBIN-Datensatz, CAD und Handskizzen).

---

## Online stellen (Cloudflare Pages)

Kostenlos, ohne Server. Die ausführliche Klick-für-Klick-Anleitung steht im Dokument „PlanDigitalizer – Nächste Schritte". Kurzfassung:

1. Den Projektordner in ein (privates) GitHub-Repository hochladen.
2. Auf <https://dash.cloudflare.com> ein kostenloses Konto anlegen → **Workers & Pages** → **Pages** → **Mit Git verbinden** → Repository wählen.
3. Einstellungen: *Framework preset* **None**, *Build command* **leer**, *Build output directory* **`web`** → Speichern und bereitstellen.
4. Nach ca. 1 Minute ist die Seite unter `https://<projektname>.pages.dev` erreichbar.

Jede Änderung im GitHub-Repository wird automatisch neu veröffentlicht.

Die Seite funktioniert auf jedem statischen Hosting (Netlify, Vercel, GitHub Pages, eigener Webspace): einfach den Inhalt des Ordners `web/` veröffentlichen. Die Datei `web/_headers` setzt auf Cloudflare Pages und Netlify zusätzliche Sicherheits-Header.

---

## Datenschutz

- **Keine Übertragung:** Pläne werden mit `FileReader` im Browser gelesen und im Arbeitsspeicher verarbeitet. Es gibt keinen Upload-Endpunkt.
- **Technisch abgesichert:** Eine Content-Security-Policy (`connect-src 'self'`) verbietet dem Browser jede Verbindung zu fremden Servern. Alle Programmteile (Python, OpenCV, Texterkennung, 3D) liegen auf der eigenen Website – kein CDN, kein Tracking, keine externen Schriften.
- **Nichts bleibt liegen:** Zwischenergebnisse existieren nur, solange der Tab offen ist; „Neuen Plan digitalisieren" löscht sie sofort.
- **Ohne Login:** Wer die Adresse kennt, kann das Werkzeug benutzen. Weil dabei keine Daten den eigenen Rechner verlassen, ist das unbedenklich. Suchmaschinen werden per `robots.txt` und `noindex` ausgeschlossen.

---

## IFC und 3D-Modell

Zusätzlich zu PDF und DXF wird aus dem Grundriss ein **vereinfachtes Gebäudemodell** abgeleitet – **nur wenn der Massstab bekannt ist** (sonst wäre das Modell wertlos).

| IFC-Element | Woher stammen die Werte |
|---|---|
| `IfcBuildingStorey` | Geschosstabelle (Einstellungen); Geschoss des Plans aus der Beschriftung („Grundriss EG") oder Auswahl |
| `IfcWall` | Lage, Länge, Stärke **gemessen**; Höhe = Einstellung; aussen/innen aus dem Gebäudeumriss |
| `IfcDoor` + Öffnung | Breite **gemessen**; Höhe = Standardwert |
| `IfcWindow` + Öffnung | Breite **gemessen**; Höhe und Brüstung = Standardwerte |
| `IfcSpace` | Fläche **berechnet**; Name aus erkanntem Text, sonst „Raum 01" … |
| `IfcSlab` | Umriss aus Aussenwänden; Stärke = Geschosshöhe − Wandhöhe |
| `IfcStair` | Lage, Breite, Auftritt **gemessen**; Steigung aus Geschosshöhe; vereinfacht |
| `IfcRoof` | **nie** – aus einem Grundriss nicht ableitbar |

Jedes Bauteil trägt den Eigenschaftssatz **`PlanDigitalizer_Herkunft`** („aus Plan erkannt" / „Annahme"). Türen und Fenster sind echte Öffnungen (`IfcRelVoidsElement`/`IfcRelFillsElement`). IFC, 3D-Modell und DXF liegen deckungsgleich (gleicher Nullpunkt; IFC/3D in Metern, DXF in mm).

**Öffnen:** IFC in Vectorworks (*Datei → Importieren → IFC*), ArchiCAD (*Ablage → Öffnen*), Revit, BIMcollab Zoom · OBJ direkt in der macOS-Vorschau · GLB in Blender, Lumion, Windows 3D-Viewer.

---

## Technik

| Aufgabe | Lösung | Warum |
|---|---|---|
| Rechnen im Browser | **Pyodide** (Python als WebAssembly) | Der bewährte Python-Kern läuft unverändert im Browser |
| Bildverarbeitung | **OpenCV** + NumPy | Standard für Perspektive, Schwellwerte, Morphologie |
| Erkennung | eigenes U-Net (PyTorch-Training, ONNX), ausgeführt mit **OpenCV-DNN** | Klein, schnell, im Browser identisch wie auf dem Desktop |
| Linien | eigene Skelettierung + Pfadverfolgung, Kreisanpassung | Polylinien *und* Bögen, robust bei Handzeichnungen |
| Texterkennung | **Tesseract.js** (Deutsch/Englisch, lokal) | Kostenlos, offline, keine Cloud |
| PDF | **PyMuPDF** | Seiten rendern + echte Vektor-PDF mit Ebenen |
| CAD | **ezdxf** → DXF | Ausgereift, Layer, Einheiten |
| IFC | eigener IFC4-Schreiber | Klein, schnell; mit IfcOpenShell auf Schemakonformität geprüft |
| 3D | **three.js** (Ansicht, GLB/OBJ-Export) | Verbreitet, läuft in jedem Browser |
| Hosting | statische Dateien (z.B. Cloudflare Pages) | Kostenlos, kein Server nötig |

---

## Grenzen und nächste Schritte

- Trainiert wurde mit Wohnungsgrundrissen. Sehr ungewöhnliche Darstellungen (z.B. Industriebau, Schnitte, Ansichten) werden schlechter erkannt.
- Schräge und runde Wände bleiben als Umriss erhalten, gehen aber nicht ins IFC.
- **Handschrift** wird kaum gelesen; Raumnamen in Handschrift fehlen.
- Ein um 90° gedrehter Plan (Hochformat ↔ Querformat) bleibt so, wie er fotografiert wurde.
- Schräg fotografierte Pläne werden entzerrt; das Seitenverhältnis kann dabei um einige Prozent abweichen (geometrisch nicht eindeutig bestimmbar). Für massgenaues Arbeiten einen Scan oder ein möglichst frontales Foto verwenden.
- IFC/3D: Treppen vereinfacht; kein Dach.
- Mehrseitige PDF: nur Seite 1.
- Sehr grosse Pläne können auf Handys am Arbeitsspeicher scheitern – dann Computer verwenden.

Sinnvolle Erweiterungen: mehrseitige PDF (ein Geschoss pro Seite) · Vektor-PDFs direkt übernehmen · schräge Wände im 3D-Modell · Nachtraining mit eigenen, anonymisierten Bestandesplänen (siehe unten).

---

## Für Entwickler

```
plandigitalizer/
├─ app/                 Python-Kern (läuft im Browser über Pyodide und lokal für Tests)
│  ├─ pipeline/         Laden → Bildkorrektur → Text → Vektorisierung → Klassifizierung → Massstab
│  ├─ export/           dxf.py, pdf.py, svg.py
│  │  ├─ semantic.py    Netz (Wand/Fenster/Tür), Ausrichtung, Perspektive
│  │  ├─ essential.py   saubere Wände, Öffnungen, Tür-/Fenstersymbole
│  │  └─ walls.py       Wandzerlegung (gemeinsam mit bim/)
│  ├─ models/           plannet.onnx (Erkennungsmodell)
│  ├─ bim/              derive.py (Gebäudemodell), ifc_writer.py
│  └─ web_api.py        Schnittstelle zum Browser-Worker
├─ ml/                  Training des Erkennungsmodells (nicht Teil der Website)
│  ├─ synth.py          Trainingspläne aus ResPlan-Geometrien in vielen Zeichenstilen
│  ├─ make_dataset.py   Datensatz erzeugen
│  ├─ model.py, train.py, export_onnx.py, evaluate.py
├─ web/                 DIE WEBSITE (dieser Ordner wird veröffentlicht)
│  ├─ index.html, style.css, app.js     Oberfläche
│  ├─ engine.js, worker.js              Rechenkern-Steuerung (Pyodide + Tesseract.js)
│  ├─ model3d.js, viewer3d.js, export3d.js   3D-Ansicht und GLB/OBJ
│  ├─ py/app.zip        gepackter Python-Kern (von tools/build_web.py erzeugt)
│  ├─ pyodide/          Python-Laufzeit + Pakete (unverändert übernommen)
│  ├─ vendor/           three.js, Tesseract.js + Sprachdaten
│  └─ _headers, robots.txt
├─ tools/build_web.py   nach Python-Änderungen ausführen
├─ tools/serve.py       lokaler Testserver (mit denselben Sicherheits-Headern)
└─ tests/               automatische Tests
```

```bash
./start.sh                               # Website lokal öffnen (http://127.0.0.1:8000)
python3 tools/build_web.py               # nach Änderungen an app/
pip install -r requirements-dev.txt      # einmalig für die Tests
python3 -m pytest -q                     # Tests (inkl. IFC-Schema-Prüfung)
python3 tests/make_samples.py            # Beispielpläne erzeugen
```

Die Tests prüfen u.a., dass PDF/DXF echte Vektoren enthalten, der Massstab auf < 2 % stimmt, gedrehte (±23°) und perspektivisch verzerrte Pläne gerade gerichtet werden, Möbel nicht als Wände erscheinen, Wandstärken und Türbreiten stimmen, die IFC-Datei schemakonform ist und `web/py/app.zip` zum aktuellen Code passt.

**Modell neu trainieren** (z.B. mit eigenen Stilen; dauert auf 2 CPU-Kernen ca. 2–3 Stunden):

```bash
pip install torch onnx shapely
git clone https://github.com/m-agour/ResPlan && cd ResPlan && unzip ResPlan.zip && cd ..
python3 ml/make_dataset.py ResPlan/ResPlan.pkl ResPlan/split.json data/synth 7000 300 300
python3 ml/train.py data/synth runs/plannet.pt --iters 16000
python3 ml/export_onnx.py runs/plannet_best.pt app/models/plannet.onnx
python3 ml/evaluate.py data/synth 150      # Messwerte auf ungesehenen Plänen
python3 tools/build_web.py
```

---

## Lizenzen und Quellen

| Komponente | Lizenz |
|---|---|
| Erkennungsmodell `plannet.onnx` | trainiert mit Geometrien aus **ResPlan** (M. Agour et al., <https://github.com/m-agour/ResPlan>, Daten CC BY 4.0) |
| Testzeichnungen (nur Prüfung, nicht verteilt) | **ROBIN** (D. Sharma et al., ICDAR 2017, <https://github.com/gesstalt/ROBIN>) |
| Pyodide, three.js, ezdxf, fonttools, pyparsing | MPL-2.0 / MIT |
| OpenCV, Tesseract / Tesseract.js, NumPy, Pillow | Apache-2.0 / BSD / HPND |
| **PyMuPDF** | **AGPL-3.0** – frei nutzbar. Wird die Website Dritten zur Verfügung gestellt, muss der Quellcode auf Anfrage zugänglich sein (z.B. über das GitHub-Repository). |
