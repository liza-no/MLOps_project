"""Project settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_NAME = "project_mlops"
PACKAGE_NAME = "project_mlops"

from project_mlops.pipeline_registry import register_pipelines
PIPELINE_REGISTRY = register_pipelines
