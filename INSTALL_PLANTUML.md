# 🌿 PlantUML Installation Guide (Windows)

Since `winget install PlantUML.PlantUML` doesn't work, here are alternative methods:

## Method 1: Download PlantUML JAR (Recommended - Easiest)

1. **Download PlantUML JAR:**
   - Go to: https://github.com/plantuml/plantuml/releases/latest
   - Download: `plantuml-1.2024.XX.jar` (latest version)
   - Save to: `C:\Users\Adhi\Desktop\stock\plantuml.jar`

2. **Install Java (if not installed):**
   ```powershell
   winget install EclipseAdoptium.Temurin-JRE
   ```
   Or download from: https://adoptium.net/

3. **Install Graphviz (required for diagrams):**
   ```powershell
   winget install Graphviz.Graphviz
   ```

4. **Create a simple batch file to use PlantUML:**
   Create `plantuml.bat` in your project folder:
   ```batch
   @echo off
   java -jar "%~dp0plantuml.jar" %*
   ```

5. **Test it:**
   ```powershell
   .\plantuml.bat architecture_diagram.puml
   ```

---

## Method 2: Use Online PlantUML (No Installation)

**Best for quick testing or if you don't want to install anything:**

1. **Go to:** https://www.plantuml.com/plantuml/uml/
2. **Copy content** from each `.puml` file
3. **Paste** into the online editor
4. **Click "Submit"** to generate PNG
5. **Download** the PNG file
6. **Save** with the correct name (e.g., `architecture_diagram.png`)

**Files to convert:**
- `architecture_diagram.puml` → `architecture_diagram.png`
- `sequence_diagram.puml` → `sequence_diagram.png`
- `training_pipeline.puml` → `training_pipeline.png`
- `recommendation_flow.puml` → `recommendation_flow.png`

---

## Method 3: Use VS Code Extension (If you use VS Code)

1. **Install VS Code extension:**
   - Open VS Code
   - Extensions → Search "PlantUML"
   - Install "PlantUML" by jebbs

2. **Right-click on `.puml` file** → "Export Current Diagram"

---

## Method 4: Use Docker (If you have Docker)

```powershell
docker run --rm -v ${PWD}:/data plantuml/plantuml *.puml
```

---

## Quick Fix: Updated Batch Script

I'll update the `generate_pdf.bat` to work with the JAR method. Just download the JAR file and place it in your project folder.

---

## Recommended Approach

**For now (fastest):**
1. Use **Method 2 (Online)** to generate the 4 PNG files
2. Save them in your project folder
3. Compile LaTeX normally

**For later (automated):**
1. Use **Method 1 (JAR)** for permanent setup

