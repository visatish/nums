# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from setuptools import setup, find_packages


requirements = [
    'numpy<=1.20.0',
    'scipy<=1.5.0',
    'ray==0.8.7',
    'boto3<=1.15.0'
]


test_requirements = [
    'pytest',
    'pytest-pylint',
]


__version__ = None


with open('nums/core/version.py') as f:
    # pylint: disable=exec-used
    exec(f.read(), globals())


with open("README.md", "r") as fh:
    long_description = fh.read()


def main():

    setup(
        name='nums',
        version=__version__,
        description="A numerical computing library for Python that scales.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/nums-project/nums",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Unix",
        ],
        python_requires='>=3.6',
        install_requires=requirements,
        test_requirements=test_requirements
    )


if __name__ == "__main__":
    main()
