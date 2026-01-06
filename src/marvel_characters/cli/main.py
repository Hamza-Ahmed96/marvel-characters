"""Command-line interface for the Marvel MLOps Teaching Agent."""

import click

from marvel_characters.teaching_agent import TeachingAgent


@click.group()
@click.option('--config', default='project_config_marvel.yml', help='Path to configuration file')
@click.option('--env', default='dev', type=click.Choice(['dev', 'acc', 'prd']), help='Environment')
@click.pass_context
def cli(ctx: click.Context, config: str, env: str) -> None:
    """Marvel MLOps Teaching Agent - Learn MLOps through hands-on examples."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['env'] = env
    try:
        ctx.obj['agent'] = TeachingAgent(config, env)
    except FileNotFoundError:
        ctx.obj['agent'] = TeachingAgent()


@cli.command()
@click.pass_context
def welcome(ctx: click.Context) -> None:
    """Display welcome message and agent overview."""
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.welcome_message())


@cli.command()
@click.pass_context
def help_guide(ctx: click.Context) -> None:
    """Show detailed help and usage guide."""
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.get_help())


@cli.group()
def concept() -> None:
    """Learn MLOps concepts."""
    pass


@concept.command('list')
@click.pass_context
def list_concepts(ctx: click.Context) -> None:
    """List all available MLOps concepts."""
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.list_concepts())


@concept.command('explain')
@click.argument('concept_name')
@click.pass_context
def explain_concept(ctx: click.Context, concept_name: str) -> None:
    """Explain a specific MLOps concept.

    CONCEPT_NAME: Name of the concept (e.g., experiment_tracking, model_registry)
    """
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.teach_concept(concept_name))


@concept.command('compare')
@click.argument('approach1')
@click.argument('approach2')
@click.pass_context
def compare_approaches(ctx: click.Context, approach1: str, approach2: str) -> None:
    """Compare two MLOps approaches.

    Examples:
        marvel-teach concept compare basic_model custom_model
        marvel-teach concept compare batch realtime
    """
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.compare_approaches(approach1, approach2))


@cli.command()
@click.pass_context
def workflow(ctx: click.Context) -> None:
    """Show the complete MLOps workflow overview."""
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.show_workflow())


@cli.group()
def review() -> None:
    """Code review and best practices."""
    pass


@review.command('file')
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def review_file(ctx: click.Context, file_path: str) -> None:
    """Review a Python file and provide feedback.

    FILE_PATH: Path to the Python file to review
    """
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.review_code_file(file_path))


@review.command('best-practices')
@click.argument('category', required=False)
@click.pass_context
def best_practices(ctx: click.Context, category: str | None) -> None:
    """Show MLOps best practices.

    CATEGORY: Optional category (mlflow_tracking, data_processing, model_training, deployment, code_quality)
    """
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.show_best_practices(category))


@cli.group()
def model() -> None:
    """Model exploration and explanations."""
    pass


@model.command('architecture')
@click.pass_context
def model_architecture(ctx: click.Context) -> None:
    """Explain the Marvel model architecture."""
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.explain_model_architecture())


@model.command('prediction-workflow')
@click.pass_context
def prediction_workflow(ctx: click.Context) -> None:
    """Explain the model prediction workflow."""
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.explain_prediction_workflow())


@model.command('demo')
@click.argument('model_uri')
@click.option('--character-type', default='superhero',
              type=click.Choice(['superhero', 'villain', 'mutant']),
              help='Type of character to create for demo')
@click.pass_context
def demo_prediction(ctx: click.Context, model_uri: str, character_type: str) -> None:
    """Demonstrate model prediction with a sample character.

    MODEL_URI: URI of the model (e.g., models:/marvel-survival-predictor/champion)

    Examples:
        marvel-teach model demo "models:/marvel-survival-predictor/1"
        marvel-teach model demo "models:/marvel-survival-predictor/champion" --character-type villain
    """
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.demo_prediction(model_uri, character_type))


@model.command('feature-importance')
@click.pass_context
def feature_importance(ctx: click.Context) -> None:
    """Explain feature importance and how to interpret it."""
    agent: TeachingAgent = ctx.obj['agent']
    if agent.model_explainer:
        click.echo(agent.model_explainer.explain_feature_importance())
    else:
        click.echo("Model explainer not available. Initialize with a config file.")


@cli.group()
def learn() -> None:
    """Learning paths and guidance."""
    pass


@learn.command('path')
@click.argument('level', type=click.Choice(['beginner', 'intermediate', 'advanced']), default='beginner')
@click.pass_context
def learning_path(ctx: click.Context, level: str) -> None:
    """Get a recommended learning path.

    LEVEL: Your current level (beginner, intermediate, advanced)
    """
    agent: TeachingAgent = ctx.obj['agent']
    click.echo(agent.get_learning_path(level))


@cli.command()
@click.argument('query')
@click.pass_context
def quick(ctx: click.Context, query: str) -> None:
    """Quick access to common queries.

    Examples:
        marvel-teach quick workflow
        marvel-teach quick help
        marvel-teach quick concepts
    """
    agent: TeachingAgent = ctx.obj['agent']

    query_map = {
        'workflow': agent.show_workflow,
        'help': agent.get_help,
        'concepts': agent.list_concepts,
        'welcome': agent.welcome_message,
    }

    if query.lower() in query_map:
        click.echo(query_map[query.lower()]())
    else:
        click.echo(f"Unknown query: {query}")
        click.echo(f"Available quick queries: {', '.join(query_map.keys())}")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
