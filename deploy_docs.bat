@echo off
REM Documentation deployment script for Entanglement-Enhanced NLP (Windows)
REM This script builds and deploys the documentation to GitHub Pages

echo 🚀 Starting documentation deployment...

REM Check if mkdocs is installed
mkdocs --version >nul 2>&1
if errorlevel 1 (
    echo ❌ MkDocs is not installed. Installing...
    pip install mkdocs mkdocs-material pymdown-extensions
)

REM Check if we're in the right directory
if not exist "mkdocs.yml" (
    echo ❌ mkdocs.yml not found. Please run this script from the project root.
    exit /b 1
)

REM Verify all documentation files exist
echo 📋 Checking documentation files...

set "files=docs\index.md docs\installation.md docs\usage.md docs\api.md docs\theory.md docs\architecture.md docs\cli.md docs\examples.md"

for %%f in (%files%) do (
    if not exist "%%f" (
        echo ❌ Missing required file: %%f
        exit /b 1
    )
)

echo ✅ All required documentation files found

REM Test the documentation build locally
echo 🔨 Testing documentation build...
mkdocs build --strict

if errorlevel 1 (
    echo ❌ Documentation build failed
    exit /b 1
) else (
    echo ✅ Documentation build successful
)

REM Deploy to GitHub Pages
echo 🌐 Deploying to GitHub Pages...
mkdocs gh-deploy --force

if errorlevel 1 (
    echo ❌ Deployment failed
    exit /b 1
) else (
    echo ✅ Documentation successfully deployed to GitHub Pages!
    echo 📖 Your documentation will be available at: https://krish567366.github.io/entanglement-enhanced-nlp/
)

echo 🎉 Documentation deployment complete!
pause
