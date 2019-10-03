#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import setup, find_packages

def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]

setup(
    name="rr",
    version="1.0.2",
    description="Basic example of a Reproducible Research Project in Python",
    url="https://gitlab.idiap.ch/software/master-ai-m05-ex6",
    license="BSD",
    author="Andre Anjos",
    author_email="andre.anjos@idiap.ch",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["rr-paper = rr.paper:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
