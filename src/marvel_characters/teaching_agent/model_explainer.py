"""Model prediction and explanation capabilities for teaching."""

from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from marvel_characters.config import ProjectConfig


class ModelExplainer:
    """Explains model predictions and demonstrates inference workflows."""

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize the model explainer.

        :param config: Project configuration
        """
        self.config = config
        self.client = MlflowClient()

    def explain_prediction_workflow(self) -> str:
        """Explain the complete prediction workflow.

        :return: Formatted explanation of prediction workflow
        """
        return """
{'='*80}
MODEL PREDICTION WORKFLOW
{'='*80}

STEP 1: PREPARE INPUT DATA
The model expects a DataFrame with these features:

Numerical Features:
- Height: Character height in cm
- Weight: Character weight in kg

Categorical Features:
- Universe: Marvel universe (e.g., 'Earth-616', 'Earth-1610')
- Identity: Secret identity status
- Gender: Character gender
- Marital_Status: Marital status
- Teams: Team affiliations
- Origin: Character origin type
- Magic: Has magical abilities (True/False)
- Mutant: Is a mutant (True/False)

STEP 2: FEATURE PREPROCESSING
The model pipeline automatically:
1. Scales numerical features (StandardScaler)
2. Encodes categorical features (OneHotEncoder)
3. Handles missing values
4. Aligns feature names

STEP 3: MODEL INFERENCE
The model makes predictions:
- Input: Preprocessed features
- Output: Binary classification (0=dead, 1=alive)
- Custom wrapper converts to human-readable format

STEP 4: OUTPUT INTERPRETATION
- "alive": Character is predicted to survive
- "dead": Character is predicted to be deceased

EXAMPLE WORKFLOW:
```python
# 1. Prepare input
character_data = pd.DataFrame([{
    'Height': 188, 'Weight': 90,
    'Universe': 'Earth-616', 'Gender': 'Male',
    'Identity': 'Public', 'Marital_Status': 'Single',
    'Teams': 'Avengers', 'Origin': 'Human',
    'Magic': False, 'Mutant': False
}])

# 2. Load model
model = mlflow.pyfunc.load_model("models:/marvel-survival-predictor/champion")

# 3. Make prediction
prediction = model.predict(character_data)
# Output: {'Survival prediction': ['alive']}
```

{'='*80}
"""

    def create_sample_character(self, character_type: str = "superhero") -> dict[str, Any]:
        """Create sample character data for testing.

        :param character_type: Type of character (superhero, villain, mutant)
        :return: Dictionary with character features
        """
        templates = {
            "superhero": {
                "Height": 188,
                "Weight": 90,
                "Universe": "Earth-616",
                "Identity": "Public",
                "Gender": "Male",
                "Marital_Status": "Single",
                "Teams": "Avengers",
                "Origin": "Human",
                "Magic": False,
                "Mutant": False
            },
            "villain": {
                "Height": 185,
                "Weight": 95,
                "Universe": "Earth-616",
                "Identity": "Secret",
                "Gender": "Male",
                "Marital_Status": "Single",
                "Teams": "None",
                "Origin": "Human",
                "Magic": True,
                "Mutant": False
            },
            "mutant": {
                "Height": 175,
                "Weight": 70,
                "Universe": "Earth-616",
                "Identity": "Public",
                "Gender": "Female",
                "Marital_Status": "Single",
                "Teams": "X-Men",
                "Origin": "Mutant",
                "Magic": False,
                "Mutant": True
            }
        }
        return templates.get(character_type, templates["superhero"])

    def explain_model_architecture(self) -> str:
        """Explain the Marvel model architecture.

        :return: Formatted explanation of model architecture
        """
        return """
{'='*80}
MARVEL MODEL ARCHITECTURE
{'='*80}

MODEL TYPE: LightGBM Classifier with Custom PyFunc Wrapper

ARCHITECTURE LAYERS:

1. PREPROCESSING PIPELINE (sklearn)
   ├─ Numerical Features → StandardScaler
   │  └─ Transforms: Height, Weight
   │      - Centers to mean=0, std=1
   │      - Prevents feature magnitude bias
   │
   └─ Categorical Features → OneHotEncoder
      └─ Transforms: Universe, Identity, Gender, etc.
          - Creates binary columns per category
          - Handles unknown categories gracefully

2. BASE MODEL (LightGBM)
   ├─ Algorithm: Gradient Boosting Decision Trees
   ├─ Hyperparameters:
   │  ├─ learning_rate: 0.01 (step size for updates)
   │  ├─ n_estimators: 1000 (number of trees)
   │  └─ max_depth: 6 (tree complexity)
   │
   └─ Output: Raw probabilities [0.0 - 1.0]

3. CUSTOM WRAPPER (MLflow PyFunc)
   ├─ Purpose: Production-ready interface
   ├─ Input: Raw character features (DataFrame)
   ├─ Processing:
   │  1. Loads sklearn pipeline
   │  2. Preprocesses features
   │  3. Gets LightGBM predictions
   │  4. Converts to human-readable format
   │
   └─ Output: {"Survival prediction": ["alive" | "dead"]}

WHY THIS ARCHITECTURE?

✓ MODULARITY
  - Preprocessing separate from model
  - Easy to update components independently

✓ PRODUCTION-READY
  - Custom wrapper handles formatting
  - Consistent input/output interface

✓ REPRODUCIBILITY
  - All components versioned together
  - MLflow tracks full pipeline

✓ PERFORMANCE
  - LightGBM is fast and accurate
  - Efficient for categorical features

TRAINING PROCESS:
1. Load training data from Unity Catalog
2. Fit preprocessing pipeline on training data
3. Transform features through pipeline
4. Train LightGBM on transformed features
5. Wrap in custom PyFunc for deployment
6. Log complete artifact to MLflow
7. Register in Model Registry

FILES TO EXPLORE:
- Pipeline: src/marvel_characters/models/basic_model.py
- Wrapper: src/marvel_characters/models/custom_model.py
- Training: scripts/train_register_custom_model.py

{'='*80}
"""

    def get_model_info(self, model_name: str) -> str:
        """Get information about a registered model.

        :param model_name: Name of the registered model
        :return: Formatted model information
        """
        try:
            model = self.client.get_registered_model(model_name)
            versions = self.client.search_model_versions(f"name='{model_name}'")

            info = f"""
MODEL INFORMATION: {model_name}
{'='*60}

Description: {model.description or 'No description provided'}
Created: {model.creation_timestamp}
Last Updated: {model.last_updated_timestamp}

VERSIONS:
"""
            for version in versions[:5]:  # Show last 5 versions
                info += f"""
  Version {version.version}:
    - Status: {version.status}
    - Created: {version.creation_timestamp}
    - Tags: {version.tags}
"""
            return info

        except Exception as e:
            return f"Error retrieving model info: {str(e)}\n\nMake sure the model '{model_name}' exists in MLflow Model Registry."

    def demonstrate_inference(self, model_uri: str, sample_type: str = "superhero") -> str:
        """Demonstrate model inference with a sample character.

        :param model_uri: URI of the model to load
        :param sample_type: Type of sample character to create
        :return: Formatted demonstration output
        """
        try:
            # Create sample data
            sample_data = self.create_sample_character(sample_type)
            sample_df = pd.DataFrame([sample_data])

            # Load and predict
            model = mlflow.pyfunc.load_model(model_uri)
            prediction = model.predict(sample_df)

            demo = f"""
INFERENCE DEMONSTRATION
{'='*60}

INPUT CHARACTER ({sample_type.upper()}):
"""
            for feature, value in sample_data.items():
                demo += f"  {feature}: {value}\n"

            demo += f"""
MODEL URI: {model_uri}

PREDICTION:
  {prediction}

INTERPRETATION:
  This {sample_type} character is predicted to be: {prediction.get('Survival prediction', ['unknown'])[0]}

CONFIDENCE NOTE:
  For probability scores, check model metadata or use predict_proba
  if available in the underlying sklearn pipeline.

{'='*60}
"""
            return demo

        except Exception as e:
            return f"Error during inference demonstration: {str(e)}\n\nCheck that:\n1. Model URI is correct\n2. MLflow tracking server is accessible\n3. Model is properly logged and registered"

    def explain_feature_importance(self) -> str:
        """Explain how to interpret feature importance.

        :return: Formatted explanation
        """
        return """
{'='*80}
UNDERSTANDING FEATURE IMPORTANCE
{'='*80}

WHAT IS FEATURE IMPORTANCE?
Feature importance tells us which character attributes most influence
the survival prediction.

HOW IS IT CALCULATED?
LightGBM calculates importance by:
- Gain: Total improvement in loss when splitting on this feature
- Split: Number of times feature is used for splitting
- Cover: Average coverage (samples affected) by splits

EXAMPLE INTERPRETATION:
If 'Magic' has high importance:
→ Magical abilities strongly correlate with survival
→ Model heavily relies on this feature
→ Changes to this feature significantly impact predictions

HOW TO ACCESS:
```python
import mlflow

# Load the sklearn pipeline from artifacts
model_uri = "runs:/<run-id>/model"
model = mlflow.sklearn.load_model(model_uri)

# Get the LightGBM model from pipeline
lgbm_model = model.named_steps['classifier']

# Get feature importances
importances = lgbm_model.feature_importances_
feature_names = model.feature_names_in_

# Display
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
```

USING IMPORTANCE FOR LEARNING:
1. Understand which features drive predictions
2. Identify potentially biased or spurious correlations
3. Guide feature engineering efforts
4. Explain model decisions to stakeholders

CAUTION:
- Importance ≠ causation
- Correlated features may split importance
- Consider SHAP values for more detailed explanations

{'='*80}
"""
