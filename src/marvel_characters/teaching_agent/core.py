"""Core Teaching Agent that orchestrates all teaching capabilities."""

from marvel_characters.config import ProjectConfig
from marvel_characters.teaching_agent.code_reviewer import CodeReviewer
from marvel_characters.teaching_agent.mlops_concepts import MLOpsConceptsTeacher
from marvel_characters.teaching_agent.model_explainer import ModelExplainer


class TeachingAgent:
    """All-in-one AI teaching agent for MLOps education.

    This agent combines multiple teaching capabilities:
    - MLOps concept explanations
    - Model prediction demonstrations
    - Code review and feedback
    - Workflow walkthroughs
    """

    def __init__(self, config_path: str | None = None, env: str = "dev") -> None:
        """Initialize the teaching agent.

        :param config_path: Path to project configuration file
        :param env: Environment to use (dev, acc, prd)
        """
        self.env = env

        # Load config if provided
        if config_path:
            self.config = ProjectConfig.from_yaml(config_path, env)
        else:
            self.config = None

        # Initialize teaching modules
        self.concepts_teacher = MLOpsConceptsTeacher()
        self.code_reviewer = CodeReviewer()

        # Initialize model explainer if config available
        if self.config:
            self.model_explainer = ModelExplainer(self.config)
        else:
            self.model_explainer = None

    def welcome_message(self) -> str:
        """Generate a welcome message for students.

        :return: Formatted welcome message
        """
        return """
{'='*80}
                    MARVELOUS MLOps TEACHING AGENT
{'='*80}

Welcome to the interactive MLOps learning assistant!

This AI agent helps you learn MLOps concepts and best practices through
the Marvel Characters project - a complete end-to-end MLOps pipeline.

WHAT CAN I HELP YOU WITH?

1. LEARN CONCEPTS
   - Understand MLOps fundamentals
   - Explore experiment tracking, model registry, deployment, etc.
   - Get hands-on examples from this project

2. EXPLORE THE WORKFLOW
   - See how a complete MLOps pipeline works
   - Understand data → training → deployment → monitoring
   - Learn about CI/CD for ML

3. REVIEW CODE
   - Get feedback on your ML code
   - Learn best practices
   - Improve code quality

4. MODEL EXPLANATIONS
   - Understand model architecture
   - See prediction workflows
   - Explore feature engineering

5. PRACTICE & EXPERIMENT
   - Try model predictions
   - Test different scenarios
   - Learn by doing

AVAILABLE COMMANDS:
  - List concepts: See all available MLOps topics
  - Explain concept: Deep dive into specific topics
  - Review code: Get feedback on your code
  - Show workflow: See the complete MLOps pipeline
  - Demo prediction: See model inference in action
  - Best practices: Learn MLOps best practices

Ready to start learning? Let's make MLOps marvelous!

{'='*80}
"""

    def teach_concept(self, concept_name: str) -> str:
        """Teach a specific MLOps concept.

        :param concept_name: Name of the concept to teach
        :return: Formatted explanation
        """
        return self.concepts_teacher.explain_concept(concept_name)

    def list_concepts(self) -> str:
        """List all available teaching concepts.

        :return: Formatted list of concepts
        """
        concepts = self.concepts_teacher.list_concepts()
        output = "\nAVAILABLE MLOps CONCEPTS\n{'='*60}\n\n"
        for i, concept in enumerate(concepts, 1):
            concept_data = self.concepts_teacher.get_concept(concept)
            output += f"{i}. {concept.upper().replace('_', ' ')}\n"
            output += f"   {concept_data['description']}\n\n"
        output += "Use 'explain <concept_name>' to learn more about any topic.\n"
        return output

    def show_workflow(self) -> str:
        """Show the complete MLOps workflow.

        :return: Formatted workflow overview
        """
        return self.concepts_teacher.get_workflow_overview()

    def review_code_file(self, file_path: str) -> str:
        """Review a code file and provide educational feedback.

        :param file_path: Path to the file to review
        :return: Formatted review
        """
        return self.code_reviewer.review_file(file_path)

    def show_best_practices(self, category: str | None = None) -> str:
        """Show MLOps best practices.

        :param category: Optional category to filter by
        :return: Formatted best practices
        """
        return self.code_reviewer.get_best_practices(category)

    def explain_model_architecture(self) -> str:
        """Explain the model architecture.

        :return: Formatted explanation
        """
        if not self.model_explainer:
            return "Model explainer not available. Please initialize with a config file."
        return self.model_explainer.explain_model_architecture()

    def explain_prediction_workflow(self) -> str:
        """Explain the prediction workflow.

        :return: Formatted explanation
        """
        if not self.model_explainer:
            return "Model explainer not available. Please initialize with a config file."
        return self.model_explainer.explain_prediction_workflow()

    def demo_prediction(self, model_uri: str, character_type: str = "superhero") -> str:
        """Demonstrate model prediction.

        :param model_uri: URI of the model to use
        :param character_type: Type of character to create
        :return: Formatted demonstration
        """
        if not self.model_explainer:
            return "Model explainer not available. Please initialize with a config file."
        return self.model_explainer.demonstrate_inference(model_uri, character_type)

    def compare_approaches(self, approach1: str, approach2: str) -> str:
        """Compare two MLOps approaches.

        :param approach1: First approach
        :param approach2: Second approach
        :return: Formatted comparison
        """
        return self.concepts_teacher.compare_approaches(approach1, approach2)

    def get_learning_path(self, level: str = "beginner") -> str:
        """Provide a recommended learning path.

        :param level: Learning level (beginner, intermediate, advanced)
        :return: Formatted learning path
        """
        paths = {
            "beginner": """
{'='*80}
LEARNING PATH: BEGINNER
{'='*80}

Welcome! Here's your recommended path to learn MLOps:

WEEK 1: FOUNDATIONS
  □ Understand what MLOps is and why it matters
  □ Learn about experiment tracking
  □ Explore the Marvel dataset
  □ Run your first model training

  Start with:
  - Concept: experiment_tracking
  - File: notebooks/lecture3.mlflow_experiment_tracking.py
  - Try: Modify hyperparameters and track results

WEEK 2: MODEL DEVELOPMENT
  □ Learn feature engineering principles
  □ Understand model training pipelines
  □ Practice with different algorithms
  □ Track and compare experiments

  Start with:
  - Concept: feature_engineering
  - File: notebooks/lecture4.train_register_basic_model.py
  - Try: Add new features and retrain

WEEK 3: MODEL MANAGEMENT
  □ Learn about model registry
  □ Understand model versioning
  □ Practice model registration
  □ Set model aliases (champion/challenger)

  Start with:
  - Concept: model_registry
  - File: src/marvel_characters/models/custom_model.py
  - Try: Register different model versions

WEEK 4: DEPLOYMENT BASICS
  □ Learn about model serving
  □ Understand API endpoints
  □ Practice model deployment
  □ Test predictions

  Start with:
  - Concept: model_serving
  - File: notebooks/lecture6.deploy_model_serving_endpoint.py
  - Try: Deploy and query your model

NEXT STEPS:
  Move to intermediate level to learn monitoring and CI/CD!

{'='*80}
""",
            "intermediate": """
{'='*80}
LEARNING PATH: INTERMEDIATE
{'='*80}

WEEK 1: MONITORING & OBSERVABILITY
  □ Learn model monitoring principles
  □ Set up inference logging
  □ Track data drift
  □ Create monitoring dashboards

  Start with:
  - Concept: monitoring
  - File: scripts/refresh_monitor.py
  - Try: Add custom monitoring metrics

WEEK 2: CI/CD FOR ML
  □ Understand CI/CD pipelines
  □ Learn automated testing
  □ Practice deployment automation
  □ Implement rollback strategies

  Start with:
  - Concept: ci_cd
  - File: .github/workflows/ci.yml
  - Try: Add new tests to the pipeline

WEEK 3: INFRASTRUCTURE AS CODE
  □ Learn Databricks Asset Bundles
  □ Manage multi-environment deployments
  □ Practice infrastructure versioning
  □ Implement environment promotion

  Start with:
  - Concept: databricks_bundles
  - File: databricks.yml
  - Try: Add new resources to bundle

WEEK 4: PRODUCTION BEST PRACTICES
  □ Implement A/B testing
  □ Practice model comparison
  □ Learn production debugging
  □ Optimize for cost and performance

  Start with:
  - Concept: model_serving (advanced)
  - File: notebooks/lecture6.ab_testing.py
  - Try: Set up champion/challenger testing

{'='*80}
""",
            "advanced": """
{'='*80}
LEARNING PATH: ADVANCED
{'='*80}

WEEK 1: ADVANCED MONITORING
  □ Implement custom metrics
  □ Set up anomaly detection
  □ Build drift detection pipelines
  □ Create automated alerting

  Focus: Building production-grade monitoring systems

WEEK 2: ADVANCED CI/CD
  □ Implement multi-stage deployments
  □ Add automated model validation
  □ Practice canary deployments
  □ Build rollback automation

  Focus: Enterprise-grade deployment pipelines

WEEK 3: MODEL GOVERNANCE
  □ Implement model approval workflows
  □ Track model lineage
  □ Document model cards
  □ Set up compliance checks

  Focus: Governance and compliance

WEEK 4: OPTIMIZATION & SCALING
  □ Optimize inference performance
  □ Implement caching strategies
  □ Practice autoscaling configuration
  □ Monitor and reduce costs

  Focus: Production optimization

CAPSTONE PROJECT:
  Build a complete MLOps pipeline from scratch using best practices

{'='*80}
"""
        }
        return paths.get(level, "Invalid level. Choose: beginner, intermediate, or advanced")

    def get_help(self) -> str:
        """Get help information about using the teaching agent.

        :return: Formatted help text
        """
        return """
{'='*80}
TEACHING AGENT HELP
{'='*80}

CONCEPT LEARNING:
  list-concepts           - Show all available MLOps concepts
  explain <concept>       - Deep dive into a specific concept
  workflow                - See the complete MLOps pipeline
  compare <a> <b>         - Compare two approaches
  learning-path <level>   - Get a recommended learning path

CODE REVIEW:
  review <file>           - Review a Python file
  best-practices          - Show all best practices
  best-practices <cat>    - Show practices for a category

MODEL EXPLORATION:
  model-architecture      - Understand the model structure
  prediction-workflow     - Learn the prediction process
  demo <model-uri>        - See a live prediction demo
  feature-importance      - Learn about feature importance

AVAILABLE CONCEPTS:
  - data_versioning       - Track and manage datasets
  - experiment_tracking   - Log ML experiments with MLflow
  - model_registry        - Manage model versions
  - feature_engineering   - Transform data into features
  - model_serving         - Deploy models as APIs
  - monitoring            - Track production performance
  - ci_cd                 - Automate testing and deployment
  - databricks_bundles    - Infrastructure as code

BEST PRACTICE CATEGORIES:
  - mlflow_tracking       - Experiment tracking best practices
  - data_processing       - Data handling best practices
  - model_training        - Training pipeline best practices
  - deployment            - Deployment best practices
  - code_quality          - Code quality best practices

LEARNING LEVELS:
  - beginner              - Start here if new to MLOps
  - intermediate          - For those with ML basics
  - advanced              - Production-focused topics

TIPS:
  • Start with the welcome message and workflow overview
  • Try hands-on: run the notebooks and scripts
  • Review existing code to learn patterns
  • Experiment with different parameters
  • Ask for help anytime!

{'='*80}
"""
