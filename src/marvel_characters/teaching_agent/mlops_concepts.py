"""MLOps concepts and educational content for teaching purposes."""

from typing import Any


class MLOpsConceptsTeacher:
    """Teaches MLOps concepts with practical examples from the Marvel project."""

    def __init__(self) -> None:
        """Initialize the MLOps concepts teacher."""
        self.concepts = {
            "data_versioning": {
                "title": "Data Versioning",
                "description": "Track and manage different versions of datasets to ensure reproducibility.",
                "why_important": "Data changes over time. Versioning ensures you can reproduce experiments and track data evolution.",
                "marvel_example": "The Marvel dataset is stored in Unity Catalog with versioning enabled, allowing us to track changes to character data.",
                "best_practices": [
                    "Use Unity Catalog or similar data catalog systems",
                    "Tag data versions with meaningful identifiers",
                    "Document data schema changes",
                    "Implement data quality checks before versioning"
                ],
                "code_example": """
# Loading versioned data from Unity Catalog
spark.read.table(f"{catalog_name}.{schema_name}.marvel_train")
# Unity Catalog automatically tracks versions and lineage
                """
            },
            "experiment_tracking": {
                "title": "Experiment Tracking with MLflow",
                "description": "Log and track ML experiments including parameters, metrics, and artifacts.",
                "why_important": "Enables comparison of different model approaches and ensures reproducibility of results.",
                "marvel_example": "All Marvel model training runs are tracked in MLflow with parameters like learning_rate, n_estimators, and metrics like accuracy.",
                "best_practices": [
                    "Log all hyperparameters",
                    "Track evaluation metrics consistently",
                    "Save model artifacts and dependencies",
                    "Use meaningful experiment and run names",
                    "Tag runs with git commit SHA for reproducibility"
                ],
                "code_example": """
# Setting up experiment tracking
mlflow.set_experiment(experiment_name="/Shared/marvel-characters")

with mlflow.start_run(run_name="lgbm-training", tags=tags):
    mlflow.log_params({"learning_rate": 0.01, "n_estimators": 1000})
    mlflow.log_metrics({"accuracy": 0.92, "f1_score": 0.89})
    mlflow.sklearn.log_model(model, "model")
                """
            },
            "model_registry": {
                "title": "Model Registry",
                "description": "Centralized storage for managing model lifecycle from development to production.",
                "why_important": "Provides version control for models, enables model governance, and facilitates deployment.",
                "marvel_example": "Marvel models are registered in MLflow Model Registry with stages (None, Staging, Production) and aliases.",
                "best_practices": [
                    "Use semantic versioning for models",
                    "Set model aliases (e.g., 'champion', 'challenger')",
                    "Add model descriptions and documentation",
                    "Implement model approval workflows",
                    "Track model lineage and dependencies"
                ],
                "code_example": """
# Registering a model
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name="marvel-survival-predictor",
    tags={"git_sha": "abc123", "branch": "main"}
)

# Setting model alias
client.set_registered_model_alias(
    name="marvel-survival-predictor",
    alias="champion",
    version=registered_model.version
)
                """
            },
            "feature_engineering": {
                "title": "Feature Engineering",
                "description": "Transform raw data into features that better represent the problem to the model.",
                "why_important": "Good features are critical for model performance. Systematic feature engineering ensures consistency.",
                "marvel_example": "Marvel character features include both numerical (Height, Weight) and categorical (Universe, Gender) attributes.",
                "best_practices": [
                    "Document feature transformations",
                    "Use feature stores for reusability",
                    "Validate feature distributions",
                    "Handle missing values consistently",
                    "Create features that are available at inference time"
                ],
                "code_example": """
# Feature engineering pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)
                """
            },
            "model_serving": {
                "title": "Model Serving",
                "description": "Deploy models as scalable REST APIs for real-time or batch predictions.",
                "why_important": "Makes models accessible to applications and users, enabling production use cases.",
                "marvel_example": "Marvel models are deployed to Databricks Model Serving endpoints with autoscaling and monitoring.",
                "best_practices": [
                    "Use managed serving platforms when possible",
                    "Implement health checks and monitoring",
                    "Version your API endpoints",
                    "Add request/response logging",
                    "Set appropriate resource limits and autoscaling"
                ],
                "code_example": """
# Deploying to model serving endpoint
endpoint_config = {
    "served_entities": [{
        "entity_name": "marvel-survival-predictor",
        "entity_version": "1",
        "workload_size": "Small",
        "scale_to_zero_enabled": True
    }]
}
client.create_endpoint(name="marvel-endpoint", config=endpoint_config)
                """
            },
            "monitoring": {
                "title": "Model Monitoring",
                "description": "Track model performance and data quality in production to detect issues early.",
                "why_important": "Models can degrade over time due to data drift, concept drift, or system issues.",
                "marvel_example": "Marvel model predictions are monitored for data drift, prediction distribution, and performance metrics.",
                "best_practices": [
                    "Monitor prediction distributions",
                    "Track feature drift",
                    "Set up alerting for anomalies",
                    "Log inference latency and errors",
                    "Compare ground truth when available"
                ],
                "code_example": """
# Setting up monitoring
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog

client = WorkspaceClient()
monitor = client.quality_monitors.create(
    table_name=f"{catalog}.{schema}.marvel_predictions",
    inference_log=MonitorInferenceLog(
        model_id_col="model_version",
        prediction_col="prediction",
        timestamp_col="timestamp"
    )
)
                """
            },
            "ci_cd": {
                "title": "CI/CD for ML",
                "description": "Automated testing and deployment pipelines for ML code and models.",
                "why_important": "Ensures code quality, automates deployment, and enables faster iteration.",
                "marvel_example": "GitHub Actions workflows run tests, build packages, and deploy to different environments (dev/acc/prd).",
                "best_practices": [
                    "Test data processing and model code",
                    "Validate model performance in CI",
                    "Use environment-specific configurations",
                    "Automate deployment after approvals",
                    "Implement rollback mechanisms"
                ],
                "code_example": """
# GitHub Actions workflow example
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: uv sync --extra test
      - run: pytest tests/
                """
            },
            "databricks_bundles": {
                "title": "Databricks Asset Bundles",
                "description": "Infrastructure-as-code for Databricks resources like jobs, models, and endpoints.",
                "why_important": "Enables version control, environment management, and reproducible deployments.",
                "marvel_example": "Marvel project uses bundles to deploy resources across dev, acc, and prd environments.",
                "best_practices": [
                    "Define all resources in YAML",
                    "Use variables for environment-specific configs",
                    "Version control bundle definitions",
                    "Test bundles in dev before production",
                    "Use bundle validation before deployment"
                ],
                "code_example": """
# databricks.yml
bundle:
  name: marvel-characters

targets:
  dev:
    mode: development
    workspace:
      host: ${workspace_host}
      profile: marvelous
                """
            }
        }

    def get_concept(self, concept_name: str) -> dict[str, Any] | None:
        """Get detailed information about a specific MLOps concept.

        :param concept_name: Name of the concept to retrieve
        :return: Dictionary containing concept details or None if not found
        """
        return self.concepts.get(concept_name)

    def list_concepts(self) -> list[str]:
        """List all available MLOps concepts.

        :return: List of concept names
        """
        return list(self.concepts.keys())

    def explain_concept(self, concept_name: str) -> str:
        """Generate a formatted explanation of an MLOps concept.

        :param concept_name: Name of the concept to explain
        :return: Formatted explanation string
        """
        concept = self.get_concept(concept_name)
        if not concept:
            return f"Concept '{concept_name}' not found. Available concepts: {', '.join(self.list_concepts())}"

        explanation = f"""
{'='*80}
{concept['title'].upper()}
{'='*80}

WHAT IS IT?
{concept['description']}

WHY IS IT IMPORTANT?
{concept['why_important']}

MARVEL PROJECT EXAMPLE:
{concept['marvel_example']}

BEST PRACTICES:
"""
        for i, practice in enumerate(concept['best_practices'], 1):
            explanation += f"{i}. {practice}\n"

        explanation += f"""
CODE EXAMPLE:
{concept['code_example']}

{'='*80}
"""
        return explanation

    def get_workflow_overview(self) -> str:
        """Explain the complete MLOps workflow in the Marvel project.

        :return: Formatted workflow overview
        """
        return """
{'='*80}
COMPLETE MLOps WORKFLOW - MARVEL PROJECT
{'='*80}

1. DATA PREPARATION
   - Load raw Marvel dataset from CSV
   - Validate and clean data
   - Split into train/test sets
   - Store in Unity Catalog with versioning

   Files: scripts/process_data.py, src/marvel_characters/data_processor.py

2. EXPERIMENT TRACKING
   - Set up MLflow experiments
   - Track multiple model training runs
   - Log parameters, metrics, and artifacts
   - Compare model performance

   Files: notebooks/lecture3.mlflow_experiment_tracking.py

3. MODEL TRAINING
   - Feature engineering (numerical + categorical)
   - Train LightGBM models
   - Evaluate on test set
   - Create custom model wrappers

   Files: scripts/train_register_custom_model.py,
          src/marvel_characters/models/

4. MODEL REGISTRATION
   - Register best models in MLflow Registry
   - Set model aliases (champion, challenger)
   - Tag with git SHA and metadata
   - Version models systematically

   Files: src/marvel_characters/models/custom_model.py

5. MODEL DEPLOYMENT
   - Deploy to Databricks Model Serving
   - Configure autoscaling and resources
   - Set up A/B testing (champion vs challenger)
   - Enable serverless endpoints

   Files: scripts/deploy_model.py,
          resources/model_deployment.yml

6. MONITORING & OBSERVABILITY
   - Track prediction distributions
   - Monitor for data drift
   - Log inference requests/responses
   - Set up alerts for anomalies

   Files: scripts/refresh_monitor.py,
          src/marvel_characters/monitoring.py

7. CI/CD AUTOMATION
   - Run tests on every commit
   - Build Python wheels
   - Deploy to dev → acc → prd
   - Use Databricks Asset Bundles

   Files: .github/workflows/, databricks.yml

8. ITERATION & IMPROVEMENT
   - Analyze monitoring results
   - Retrain with new data
   - Update features
   - Deploy improved models

{'='*80}
"""

    def compare_approaches(self, approach1: str, approach2: str) -> str:
        """Compare two MLOps approaches or tools.

        :param approach1: First approach to compare
        :param approach2: Second approach to compare
        :return: Comparison explanation
        """
        comparisons = {
            ("basic_model", "custom_model"): """
BASIC MODEL vs CUSTOM MODEL WRAPPER

BASIC MODEL (sklearn Pipeline):
✓ Simple and straightforward
✓ Good for quick prototyping
✓ Standard sklearn interface
✗ Limited customization
✗ Less control over output format

CUSTOM MODEL (MLflow PyFunc):
✓ Full control over predictions
✓ Custom output formatting
✓ Can include business logic
✓ Better for production
✗ More complex implementation

MARVEL PROJECT USAGE:
- Basic: lecture4.train_register_basic_model.py
- Custom: lecture4.train_register_custom_model.py

RECOMMENDATION: Start with basic for learning, use custom for production.
            """,
            ("batch", "realtime"): """
BATCH INFERENCE vs REAL-TIME SERVING

BATCH INFERENCE:
✓ Process large datasets efficiently
✓ Lower cost for bulk predictions
✓ Can use larger models
✗ Higher latency (minutes to hours)
✗ Not suitable for interactive use

REAL-TIME SERVING:
✓ Low latency (milliseconds)
✓ Interactive applications
✓ Immediate feedback
✗ Higher cost per prediction
✗ Need to manage endpoints

MARVEL PROJECT USAGE:
Real-time serving via Databricks Model Serving endpoints

WHEN TO USE EACH:
- Batch: Daily/weekly scoring of all characters
- Real-time: Interactive web app for character predictions
            """
        }

        key = (approach1.lower(), approach2.lower())
        reverse_key = (approach2.lower(), approach1.lower())

        if key in comparisons:
            return comparisons[key]
        elif reverse_key in comparisons:
            return comparisons[reverse_key]
        else:
            return f"Comparison between '{approach1}' and '{approach2}' not available."
