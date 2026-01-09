"""Installation script for the 'robotic_world_model_lite' python package."""

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "torch>=2.7",
    "rsl-rl-lib @ git+https://github.com/leggedrobotics/rsl_rl_rwm.git@main",
    "wandb",
    "pandas",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Installation operation
setup(
    name="rwm_lite",
    author="Chenhao Li",
    maintainer="Chenhao Li",
    url="https://github.com/leggedrobotics/robotic_world_model_lite",
    version=0.1,
    description="The lightweight robotic world model for quick prototyping.",
    keywords=["robotic world model", "reinforcement learning"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
