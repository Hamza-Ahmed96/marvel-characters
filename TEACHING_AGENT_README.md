# Marvel MLOps Teaching Agent - Quick Start

## What is This?

An AI-powered CLI tool that teaches you MLOps concepts through the Marvel Characters project. It combines concept explanations, code review, and hands-on demonstrations.

## Installation

Already installed! The `marvel-teach` command is now available.

## Quick Commands

### Get Started
```bash
marvel-teach welcome          # See what the agent can do
marvel-teach help-guide       # Detailed help
```

### Learn Concepts
```bash
marvel-teach concept list                         # See all topics
marvel-teach concept explain experiment_tracking  # Learn about MLflow
marvel-teach concept explain model_serving        # Learn deployment
marvel-teach concept compare basic_model custom_model  # Compare approaches
```

### View the MLOps Workflow
```bash
marvel-teach workflow         # See the complete pipeline
```

### Review Code
```bash
marvel-teach review file scripts/train_register_custom_model.py
marvel-teach review best-practices mlflow_tracking
```

### Explore the Model
```bash
marvel-teach model architecture        # Understand the model
marvel-teach model prediction-workflow # Learn inference
marvel-teach model feature-importance  # Feature importance
```

### Get a Learning Path
```bash
marvel-teach learn path beginner      # 4-week beginner path
marvel-teach learn path intermediate  # 4-week intermediate path
```

## Available Concepts

1. **data_versioning** - Track and manage datasets
2. **experiment_tracking** - Log experiments with MLflow
3. **model_registry** - Manage model versions
4. **feature_engineering** - Transform data into features
5. **model_serving** - Deploy models as APIs
6. **monitoring** - Track production performance
7. **ci_cd** - Automate testing and deployment
8. **databricks_bundles** - Infrastructure as code

## Example Workflow

**Day 1: Learn the Basics**
```bash
# Start here
marvel-teach welcome
marvel-teach concept list

# Learn your first concept
marvel-teach concept explain experiment_tracking

# Review some code
marvel-teach review file notebooks/lecture3.mlflow_experiment_tracking.py
```

**Day 2: Explore the Pipeline**
```bash
# See the big picture
marvel-teach workflow

# Understand the model
marvel-teach model architecture
marvel-teach model prediction-workflow
```

**Day 3: Learn Best Practices**
```bash
# Review best practices
marvel-teach review best-practices

# Review specific files
marvel-teach review file scripts/train_register_custom_model.py
marvel-teach review file src/marvel_characters/models/custom_model.py
```

**Day 4: Get a Structured Path**
```bash
# Get a learning path for your level
marvel-teach learn path beginner
```

## Tips

1. **Start with concepts** before diving into code
2. **Review actual project files** to see patterns
3. **Follow the learning paths** for structured learning
4. **Use help-guide** anytime you're stuck
5. **Compare approaches** to understand trade-offs

## Need Help?

```bash
marvel-teach help-guide    # Detailed help
marvel-teach quick help    # Quick help access
```

## What Makes This Special?

- **Real Production Code**: Learn from actual MLOps implementation
- **All-in-One**: Concepts + code review + demonstrations
- **Interactive**: Ask questions, explore, learn by doing
- **Progressive**: Beginner → Intermediate → Advanced paths
- **Best Practices**: Industry-standard patterns and techniques

Happy learning!
