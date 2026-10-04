# PlanDigitalizer

Online-Werkzeug zur **Digitalisierung alter Baupläne und Handzeichnungen**.

**Website öffnen → Plan hochladen → automatische Vektorisierung → PDF / DXF / IFC / 3D herunterladen → in CAD oder BIM weiterbearbeiten.**

Die gesamte Verarbeitung läuft **im Browser**. Pläne werden nicht hochgeladen, nicht gespeichert und an niemanden übertragen – auch nicht an den Webserver. Die Website braucht deshalb keinen Server, keine Datenbank und keinen Login.

---

## Inhalt

1. [Was das Werkzeug kann](#was-das-werkzeug-kann)
2. [Online stellen (Cloudflare Pages)](#online-stellen-cloudflare-pages)
3. [Datenschutz](#datenschutz)
4. [IFC und 3D-Modell](#ifc-und-3d-modell)
5. [Technik](#technik)
6. [Grenzen und nächste Schritte](#grenzen-und-nächste-schritte)
7. [Für Entwickler: lokal testen, Tests, Aufbau](#für-entwickler)
8. [Lizenzen](#lizenzen)

---

## Was das Werkzeug kann

| Bereich | Umfang |
|---|---|
| **Eingabe** | JPG, PNG, WEBP, TIFF, PDF (erste Seite) – Scan, Handyfoto, Handskizze |
| **Bildkorrektur** | Perspektive (Blatt auf Tisch), Schräglage, Schatten/Hintergrund, Kontrast, Rauschen, Schwarz-Weiss, Lückenschluss |
| **Geometrie** | Linien (waagrecht/senkrecht exakt), schräge Linien, Kreise, Bögen; Ecken und T-Stösse geschlossen |
| **Plan-Elemente** | Wände (gefüllt oder Doppellinie), Türen, Fenster, Treppen, Bemassung, Schraffuren, Texte, Symbole |
| **Texterkennung** | Deutsch + Englisch, auch senkrechte Masszahlen → editierbarer CAD-Text |
| **Massstab** | Kalibrierung (zwei Punkte + bekannte Länge) · automatisch aus Massketten · „M 1:100" + Papiergrösse |
| **Ausgabe** | Vektor-PDF (ein-/ausblendbare Ebenen) · DXF (Layer, mm) · IFC4 · 3D als GLB und OBJ |

**CAD-Layer:** `WALLS`, `DOORS`, `WINDOWS`, `STAIRS`, `TEXT`, `DIMENSIONS`, `LINES`, `HATCH`, `SYMBOLS`

**DWG:** Echtes DWG lässt sich im Browser nicht erzeugen (das Format ist proprietär; der kostenlose ODA-Konverter ist ein Desktop-Programm). Die DXF öffnet sich in jedem CAD und lässt sich dort als DWG speichern.

**Dauer:** Erster Besuch ca. 60 MB Download (danach aus dem Browser-Zwischenspeicher, Start in wenigen Sekunden). Ein Plan braucht je nach Grösse und Rechner 15–40 Sekunden.

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
| Linien | eigene Skelettierung + Pfadverfolgung, Kreisanpassung | Polylinien *und* Bögen, robust bei Handzeichnungen |
| Texterkennung | **Tesseract.js** (Deutsch/Englisch, lokal) | Kostenlos, offline, keine Cloud |
| PDF | **PyMuPDF** | Seiten rendern + echte Vektor-PDF mit Ebenen |
| CAD | **ezdxf** → DXF | Ausgereift, Layer, Einheiten |
| IFC | eigener IFC4-Schreiber | Klein, schnell; mit IfcOpenShell auf Schemakonformität geprüft |
| 3D | **three.js** (Ansicht, GLB/OBJ-Export) | Verbreitet, läuft in jedem Browser |
| Hosting | statische Dateien (z.B. Cloudflare Pages) | Kostenlos, kein Server nötig |

---

## Grenzen und nächste Schritte

- Klassifizierung ist **regelbasiert** – ungewöhnliche Zeichenstile landen auf `LINES`.
- **Handschrift** wird kaum erkannt; solche Texte bleiben als Linien erhalten.
- Massstriche an dichten Kreuzungen gehen teilweise verloren.
- IFC/3D: nur achsparallele Wände; Treppen vereinfacht; kein Dach.
- Mehrseitige PDF: nur Seite 1.
- Sehr grosse Pläne können auf Handys am Arbeitsspeicher scheitern – dann Computer verwenden.

Sinnvolle Erweiterungen: mehrseitige PDF (ein Geschoss pro Seite) · Vektor-PDFs direkt übernehmen · Regler „Detailgrad" · schräge Wände im 3D-Modell.

---

## Für Entwickler

```
plandigitalizer/
├─ app/                 Python-Kern (läuft im Browser über Pyodide und lokal für Tests)
│  ├─ pipeline/         Laden → Bildkorrektur → Text → Vektorisierung → Klassifizierung → Massstab
│  ├─ export/           dxf.py, pdf.py, svg.py
│  ├─ bim/              derive.py (Gebäudemodell), ifc_writer.py
│  └─ web_api.py        Schnittstelle zum Browser-Worker
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

Die Tests prüfen u.a., dass PDF/DXF echte Vektoren enthalten, der Massstab auf < 2 % stimmt, die IFC-Datei schemakonform ist und `web/py/app.zip` zum aktuellen Code passt.

---

## Lizenzen

| Komponente | Lizenz |
|---|---|
| Pyodide, three.js, ezdxf, fonttools, pyparsing | MPL-2.0 / MIT |
| OpenCV, Tesseract / Tesseract.js, NumPy, Pillow | Apache-2.0 / BSD / HPND |
| **PyMuPDF** | **AGPL-3.0** – frei nutzbar. Wird die Website Dritten zur Verfügung gestellt, muss der Quellcode auf Anfrage zugänglich sein (z.B. über das GitHub-Repository). |
