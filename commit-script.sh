#!/bin/bash
# Script to update the Peso GitHub repository with the reorganized structure

# Clone the repository (if not already cloned)
if [ ! -d "Peso" ]; then
  echo "Cloning repository..."
  git clone https://github.com/o0Praiz/Peso.git
  cd Peso
else
  echo "Repository already exists, updating..."
  cd Peso
  git pull
fi

# Create the directory structure
echo "Creating directory structure..."
mkdir -p database/
mkdir -p query/
mkdir -p insights/
mkdir -p collection/
mkdir -p integration/
mkdir -p ml/

# Create __init__.py files
echo "Creating __init__.py files..."
echo 'from database.database_module import PesoDatabase\n\n__all__ = ["PesoDatabase"]' > database/__init__.py
echo 'from query.query_module import QueryEngine\n\n__all__ = ["QueryEngine"]' > query/__init__.py
echo 'from insights.insights_module import InsightsEngine\n\n__all__ = ["InsightsEngine"]' > insights/__init__.py
echo 'from collection.data_collection import DataCollector\n\n__all__ = ["DataCollector"]' > collection/__init__.py
echo 'from integration.ai_integration import AIToolIntegration\n\n__all__ = ["AIToolIntegration"]' > ml/__init__.py
echo 'from ml.ml_prediction import MLPredictionEngine\n\n__all__ = ["MLPredictionEngine"]' > integration/__init__.py

# Create/update module files
echo "Creating module files..."

# Copy content from artifacts to files (assuming files exist in parent directory)
cp ../database_module.py database/
cp ../query_module.py query/
cp ../insights_module.py insights/
cp ../data_collection.py collection/
cp ../ai_integration.py integration/
cp ../ml_prediction.py ml/
cp ../main.py ./
cp ../setup.py ./
cp ../requirements.txt ./
cp ../README.md ./
cp ../Project_Progress.md ./

# Remove redundant files
echo "Removing redundant files..."
rm -f peso-insights-generation*.py
rm -f insights-generation-module.py
rm -f peso-project-overview.md
rm -f peso-project-progress.md
rm -f peso-repository-structure.sh

# Commit and push changes
echo "Committing changes..."
git add .
git commit -m "Reorganize project structure, consolidate modules, and enhance documentation"
git push origin master

echo "Repository update complete!"
