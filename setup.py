from setuptools import setup, find_packages
from setuptools.command.install import install
import os

NAME = "jax_hpc"
VERSION = "0.0"


class InstallScript(install):
    def run(self):
        install.run(self)


with open("requirements.txt", "r") as f:
    install_requires = list(
        filter(lambda x: "#" not in x, (line.strip() for line in f))
    )

setup(
    name=NAME,
    version=VERSION,
    author="Nick Charron",
    license="MIT",
    author_email="charron.nicholas.e@gmail.com",
    url="https://github.com/ruunyox/jax-hpc",
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe=True,
    entry_points={"console_scripts": ["jaxhpc = jax_hpc.scripts.__main__:main"]},
    cmdclass={"install": InstallScript},
)
