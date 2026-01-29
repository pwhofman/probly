=========
Changelog
=========

[2026-01-20] - Dokumentationsformat
==============================================

Dokumentation von Markdown zu reStructuredText
--------------------------------------------------------

**Hauptänderungen:**

- Alle Dokumentationsdateien von Markdown (.md) zu reStructuredText (.rst) konvertiert
- Code-Blöcke mit ``.. code-block::`` Direktive aktualisiert
- Tabellen zu ``list-table`` Format konvertiert
- Inline-Code zu doppelten Backticks (````code````) geändert
- Überschriften-Hierarchie an RST-Standard angepasst
- Datei-Referenzen von .md zu .rst aktualisiert

**Konvertierte Dateien:**

- ``README.md`` → ``README.rst``
- ``docs/data_generation_guide.md`` → ``docs/data_generation_guide.rst``
- ``docs/api_reference.md`` → ``docs/api_reference.rst``
- ``docs/multi_framework_guide.md`` → ``docs/multi_framework_guide.rst``

**Beibehaltene Dateien:**

- ``examples/first_order_tutorial.ipynb`` - Jupyter Notebook (keine Änderung)
- ``examples/simple_usage.py`` - Python Beispielskript (keine Änderung)


[2026-01-11] - Ursprüngliche Dokumentation
===========================================

Dokumentation erstellt und aktualisiert
----------------------------------------

**Hauptänderungen:**

- README konsolidiert (2 Dateien zusammengeführt)
- Multi-Framework Support dokumentiert (PyTorch, JAX, TensorFlow)
- API Referenz erweitert
- Benutzerhandbuch mit erweiterten Beispielen aktualisiert
- Tutorial Notebook Imports aktualisiert

**Neue Dateien:**

- README.md
- api_reference.md - Vollständige API Referenz für alle Frameworks
- multi_framework_guide.md - Framework-spezifische Implementierungsdetails
- data_generation_guide.md - Aktualisiertes Benutzerhandbuch mit Beispielen
- first_order_tutorial.ipynb - Aktualisiertes interaktives Tutorial
- simple_usage.py - Aktualisiertes Beispielskript
