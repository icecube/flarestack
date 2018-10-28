import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flarestack",
    version="1.0.2",
    author="Robert Stein",
    author_email="robert.stein@desy.de",
    description="Package for unbinned likelihood analysis of physics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="astroparticle physics science unbinned likelihood IceCube",
    url="https://github.com/robertdstein/flarestack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "numexpr",
        "healpy==1.10.3",
        "scipy",
        "matplotlib",
        "astropy"
    ],
)
