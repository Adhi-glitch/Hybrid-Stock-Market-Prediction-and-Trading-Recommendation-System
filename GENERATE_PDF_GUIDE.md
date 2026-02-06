# 📄 Step-by-Step Guide: Generate Your Research Paper PDF

## Prerequisites Check

First, check if you have the required tools installed.

### Step 1: Check if Java is installed (required for PlantUML)
```powershell
java -version
```
If you see a version number, you're good. If not, install Java:
- Download from: https://www.java.com/download/
- Or use: `winget install EclipseAdoptium.Temurin-JRE`

### Step 2: Check if Graphviz is installed (required for PlantUML)
```powershell
dot -version
```
If you see version info, you're good. If not, install:
```powershell
winget install Graphviz.Graphviz
```

### Step 3: Check if PlantUML is installed
```powershell
plantuml -version
```
If you see version info, you're good. If not, install:
```powershell
winget install PlantUML.PlantUML
```

### Step 4: Check if LaTeX (MiKTeX) is installed
```powershell
pdflatex --version
```
If you see version info, you're good. If not, install:
```powershell
winget install MiKTeX.MiKTeX
```
**Note:** MiKTeX will auto-install missing packages on first compile.

---

## Generate All Diagrams

### Step 5: Navigate to your project folder
```powershell
cd C:\Users\Adhi\Desktop\stock
```

### Step 6: Generate all PlantUML diagrams
Run these commands one by one:

```powershell
plantuml architecture_diagram.puml
plantuml sequence_diagram.puml
plantuml training_pipeline.puml
plantuml recommendation_flow.puml
```

**Expected output:** You should see 4 PNG files created:
- `architecture_diagram.png`
- `sequence_diagram.png`
- `training_pipeline.png`
- `recommendation_flow.png`

**Troubleshooting:**
- If you get "command not found", make sure PlantUML is in your PATH
- If diagrams fail, check that Java and Graphviz are installed
- You can also use an online PlantUML renderer: https://www.plantuml.com/plantuml/uml/

---

## Compile LaTeX to PDF

### Step 7: Compile the LaTeX document (first pass)
```powershell
pdflatex IEEE_Research_Paper.tex
```

**What happens:**
- LaTeX processes your document
- May prompt to install missing packages (answer "Yes")
- Creates `IEEE_Research_Paper.aux` and other auxiliary files
- May show warnings (these are usually fine)

### Step 8: Compile again (second pass - resolves references)
```powershell
pdflatex IEEE_Research_Paper.tex
```

**Why twice?**
- First pass: LaTeX processes content and creates references
- Second pass: LaTeX resolves all cross-references and citations

### Step 9: Check for your PDF
```powershell
dir IEEE_Research_Paper.pdf
```

You should see: `IEEE_Research_Paper.pdf` in your folder!

---

## Quick One-Command Script (Optional)

Create a file `generate_pdf.bat` with this content:

```batch
@echo off
echo Generating diagrams...
plantuml architecture_diagram.puml
plantuml sequence_diagram.puml
plantuml training_pipeline.puml
plantuml recommendation_flow.puml
echo.
echo Compiling LaTeX (pass 1)...
pdflatex IEEE_Research_Paper.tex
echo.
echo Compiling LaTeX (pass 2)...
pdflatex IEEE_Research_Paper.tex
echo.
echo Done! Check IEEE_Research_Paper.pdf
pause
```

Then just run:
```powershell
.\generate_pdf.bat
```

---

## Alternative: Use Overleaf (Online, No Installation)

If you prefer not to install anything locally:

1. **Go to:** https://www.overleaf.com
2. **Sign up** (free account)
3. **Create new project** → Upload Project → Upload ZIP
4. **Zip your folder:**
   ```powershell
   cd C:\Users\Adhi\Desktop
   Compress-Archive -Path stock -DestinationPath stock.zip
   ```
5. **Upload** `stock.zip` to Overleaf
6. **Generate diagrams first:**
   - Use online PlantUML: https://www.plantuml.com/plantuml/uml/
   - Download each PNG
   - Upload to Overleaf project
7. **Compile** in Overleaf (click "Recompile" button)

---

## Troubleshooting

### Problem: "PlantUML not found"
**Solution:** Add PlantUML to PATH or use full path:
```powershell
& "C:\Program Files\PlantUML\plantuml.jar" architecture_diagram.puml
```

### Problem: "Missing package" during LaTeX compile
**Solution:** MiKTeX will prompt to install. Answer "Yes" to install packages automatically.

### Problem: "Figure not found"
**Solution:** Make sure all 4 PNG files are in the same folder as `IEEE_Research_Paper.tex`

### Problem: Diagrams look wrong
**Solution:** 
- Check PlantUML syntax in the `.puml` files
- Try online renderer: https://www.plantuml.com/plantuml/uml/
- Make sure Graphviz is installed

### Problem: PDF has errors or warnings
**Solution:**
- Most warnings are harmless
- Check the `.log` file for details
- Missing references usually fix on second compile

---

## Expected Final Output

Your folder should contain:
- ✅ `IEEE_Research_Paper.pdf` (your final paper!)
- ✅ `architecture_diagram.png`
- ✅ `sequence_diagram.png`
- ✅ `training_pipeline.png`
- ✅ `recommendation_flow.png`
- ✅ `IEEE_Research_Paper.aux`, `.log`, `.out` (LaTeX auxiliary files - safe to ignore)

---

## Summary (Quick Commands)

```powershell
# Navigate to folder
cd C:\Users\Adhi\Desktop\stock

# Generate diagrams
plantuml *.puml

# Compile LaTeX
pdflatex IEEE_Research_Paper.tex
pdflatex IEEE_Research_Paper.tex

# Open PDF
start IEEE_Research_Paper.pdf
```

**Done!** 🎉

