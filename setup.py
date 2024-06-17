from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    author="Santiago García Gil",
    packages=find_packages(where="src"),
    requires=["gymnasium",
              "torch",
              "numpy",
              "pandas",
              "torchvision",
              "matplotlib",
              ]
)
