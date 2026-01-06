"""Code review and feedback functionality for teaching purposes."""

import ast
from pathlib import Path
from typing import Any


class CodeReviewer:
    """Reviews code and provides educational feedback on MLOps best practices."""

    def __init__(self) -> None:
        """Initialize the code reviewer."""
        self.best_practices = {
            "mlflow_tracking": [
                "Always set experiment name before starting runs",
                "Use meaningful run names with timestamps or descriptions",
                "Log all hyperparameters consistently",
                "Include git SHA in tags for reproducibility",
                "Log metrics at each important step",
                "Save model artifacts with clear naming"
            ],
            "data_processing": [
                "Validate input data schema before processing",
                "Handle missing values explicitly",
                "Document feature engineering transformations",
                "Use consistent train/test splits with random seeds",
                "Store processed data in version-controlled catalogs",
                "Log data statistics and distributions"
            ],
            "model_training": [
                "Use configuration files for hyperparameters",
                "Implement cross-validation for robust evaluation",
                "Track multiple metrics (not just accuracy)",
                "Save preprocessing pipelines with models",
                "Use random seeds for reproducibility",
                "Handle class imbalance if present"
            ],
            "deployment": [
                "Test models before deployment",
                "Use model registry with proper versioning",
                "Implement health checks for endpoints",
                "Set appropriate resource limits",
                "Enable monitoring from day one",
                "Document API schemas and examples"
            ],
            "code_quality": [
                "Write docstrings for all functions/classes",
                "Use type hints for better code clarity",
                "Follow PEP 8 style guidelines",
                "Add error handling for external calls",
                "Write unit tests for core logic",
                "Use logging instead of print statements"
            ]
        }

    def review_mlflow_code(self, code: str) -> dict[str, Any]:
        """Review MLflow-related code and provide feedback.

        :param code: Python code string to review
        :return: Dictionary with feedback and suggestions
        """
        issues = []
        suggestions = []
        good_practices = []

        # Check for experiment setup
        if "mlflow.set_experiment" in code:
            good_practices.append("✓ Sets experiment name explicitly")
        else:
            issues.append("Missing mlflow.set_experiment() - experiment name should be set")
            suggestions.append("Add: mlflow.set_experiment(experiment_name='/Shared/your-experiment')")

        # Check for tags
        if "tags=" in code and "mlflow.start_run" in code:
            good_practices.append("✓ Uses tags in MLflow runs")
        else:
            issues.append("No tags found in mlflow.start_run()")
            suggestions.append("Add tags with git SHA: tags={'git_sha': git_sha, 'branch': branch}")

        # Check for parameter logging
        if "mlflow.log_param" in code:
            good_practices.append("✓ Logs parameters")
        else:
            issues.append("Parameters should be logged with mlflow.log_params()")
            suggestions.append("Add: mlflow.log_params({'learning_rate': lr, 'n_estimators': n})")

        # Check for metric logging
        if "mlflow.log_metric" in code:
            good_practices.append("✓ Logs metrics")
        else:
            issues.append("Metrics should be logged with mlflow.log_metrics()")
            suggestions.append("Add: mlflow.log_metrics({'accuracy': acc, 'f1_score': f1})")

        # Check for model logging
        if "mlflow.sklearn.log_model" in code or "mlflow.pyfunc.log_model" in code:
            good_practices.append("✓ Logs model artifacts")
        else:
            issues.append("Model should be logged to MLflow")
            suggestions.append("Add: mlflow.sklearn.log_model(model, 'model')")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "good_practices": good_practices,
            "category": "mlflow_tracking"
        }

    def review_data_processing_code(self, code: str) -> dict[str, Any]:
        """Review data processing code and provide feedback.

        :param code: Python code string to review
        :return: Dictionary with feedback and suggestions
        """
        issues = []
        suggestions = []
        good_practices = []

        # Check for data validation
        if "assert" in code or "validate" in code.lower() or "schema" in code.lower():
            good_practices.append("✓ Includes data validation")
        else:
            issues.append("No data validation found")
            suggestions.append("Add schema validation or assertion checks")

        # Check for missing value handling
        if "dropna" in code or "fillna" in code or "isna" in code:
            good_practices.append("✓ Handles missing values")
        else:
            issues.append("Missing value handling not explicit")
            suggestions.append("Explicitly handle NaN values with dropna() or fillna()")

        # Check for random seed
        if "random_state" in code or "seed" in code:
            good_practices.append("✓ Uses random seed for reproducibility")
        else:
            issues.append("No random seed found")
            suggestions.append("Set random_state parameter for reproducible splits")

        # Check for data splitting
        if "train_test_split" in code:
            good_practices.append("✓ Splits data into train/test")
        else:
            issues.append("Should split data for proper evaluation")
            suggestions.append("Use: train_test_split(X, y, test_size=0.2, random_state=42)")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "good_practices": good_practices,
            "category": "data_processing"
        }

    def review_model_code(self, code: str) -> dict[str, Any]:
        """Review model training code and provide feedback.

        :param code: Python code string to review
        :return: Dictionary with feedback and suggestions
        """
        issues = []
        suggestions = []
        good_practices = []

        # Check for configuration usage
        if "config" in code.lower() or "yaml" in code.lower():
            good_practices.append("✓ Uses configuration for parameters")
        else:
            issues.append("Hardcoded parameters found")
            suggestions.append("Move hyperparameters to config file for easier tuning")

        # Check for pipeline usage
        if "Pipeline" in code:
            good_practices.append("✓ Uses sklearn Pipeline")
        else:
            suggestions.append("Consider using sklearn Pipeline to bundle preprocessing and model")

        # Check for cross-validation
        if "cross_val" in code or "StratifiedKFold" in code:
            good_practices.append("✓ Implements cross-validation")
        else:
            suggestions.append("Consider adding cross-validation for more robust evaluation")

        # Check for multiple metrics
        metrics_found = sum([
            "accuracy" in code.lower(),
            "precision" in code.lower(),
            "recall" in code.lower(),
            "f1" in code.lower(),
            "roc_auc" in code.lower()
        ])
        if metrics_found >= 2:
            good_practices.append(f"✓ Tracks multiple metrics ({metrics_found} found)")
        else:
            issues.append("Only one or no metrics found")
            suggestions.append("Track multiple metrics: accuracy, precision, recall, F1, ROC-AUC")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "good_practices": good_practices,
            "category": "model_training"
        }

    def review_code_quality(self, code: str) -> dict[str, Any]:
        """Review code quality and structure.

        :param code: Python code string to review
        :return: Dictionary with feedback and suggestions
        """
        issues = []
        suggestions = []
        good_practices = []

        try:
            tree = ast.parse(code)

            # Check for docstrings
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if functions:
                functions_with_docstrings = sum(
                    1 for func in functions
                    if ast.get_docstring(func) is not None
                )
                if functions_with_docstrings == len(functions):
                    good_practices.append("✓ All functions have docstrings")
                elif functions_with_docstrings > 0:
                    issues.append(f"Only {functions_with_docstrings}/{len(functions)} functions have docstrings")
                    suggestions.append("Add docstrings to all functions")
                else:
                    issues.append("No function docstrings found")
                    suggestions.append("Add descriptive docstrings with :param and :return tags")

            # Check for type hints
            functions_with_hints = sum(
                1 for func in functions
                if func.returns is not None or any(arg.annotation for arg in func.args.args)
            )
            if functions_with_hints > 0:
                good_practices.append(f"✓ Uses type hints ({functions_with_hints} functions)")
            else:
                suggestions.append("Add type hints for better code clarity: def func(x: int) -> str:")

        except SyntaxError:
            issues.append("Code has syntax errors")

        # Check for logging
        if "import logging" in code or "from loguru import logger" in code:
            good_practices.append("✓ Uses logging")
        else:
            if "print(" in code:
                issues.append("Uses print() instead of logging")
                suggestions.append("Replace print() with proper logging")

        # Check for error handling
        if "try:" in code and "except" in code:
            good_practices.append("✓ Includes error handling")
        else:
            suggestions.append("Add try/except blocks for external API calls and file operations")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "good_practices": good_practices,
            "category": "code_quality"
        }

    def review_file(self, file_path: str) -> str:
        """Review a Python file and provide comprehensive feedback.

        :param file_path: Path to the Python file to review
        :return: Formatted review output
        """
        try:
            with open(file_path) as f:
                code = f.read()

            # Run all reviews
            reviews = [
                self.review_mlflow_code(code),
                self.review_data_processing_code(code),
                self.review_model_code(code),
                self.review_code_quality(code)
            ]

            # Format output
            output = f"""
{'='*80}
CODE REVIEW: {Path(file_path).name}
{'='*80}

"""
            for review in reviews:
                if review["good_practices"] or review["issues"]:
                    output += f"\n{review['category'].upper().replace('_', ' ')}\n{'-'*60}\n"

                    if review["good_practices"]:
                        output += "\nGOOD PRACTICES:\n"
                        for practice in review["good_practices"]:
                            output += f"  {practice}\n"

                    if review["issues"]:
                        output += "\nISSUES TO ADDRESS:\n"
                        for issue in review["issues"]:
                            output += f"  ⚠ {issue}\n"

                    if review["suggestions"]:
                        output += "\nSUGGESTIONS:\n"
                        for suggestion in review["suggestions"]:
                            output += f"  → {suggestion}\n"

            output += f"\n{'='*80}\n"
            output += "\nRELEVANT BEST PRACTICES:\n"
            for category, practices in self.best_practices.items():
                output += f"\n{category.upper().replace('_', ' ')}:\n"
                for practice in practices:
                    output += f"  • {practice}\n"

            output += f"\n{'='*80}\n"

            return output

        except FileNotFoundError:
            return f"File not found: {file_path}"
        except Exception as e:
            return f"Error reviewing file: {str(e)}"

    def get_best_practices(self, category: str | None = None) -> str:
        """Get best practices for a specific category or all categories.

        :param category: Optional category name
        :return: Formatted best practices
        """
        if category and category in self.best_practices:
            practices = self.best_practices[category]
            output = f"\nBEST PRACTICES: {category.upper().replace('_', ' ')}\n{'='*60}\n"
            for i, practice in enumerate(practices, 1):
                output += f"{i}. {practice}\n"
            return output
        else:
            output = "\nMLOPS BEST PRACTICES\n{'='*60}\n"
            for cat, practices in self.best_practices.items():
                output += f"\n{cat.upper().replace('_', ' ')}:\n"
                for practice in practices:
                    output += f"  • {practice}\n"
            return output
