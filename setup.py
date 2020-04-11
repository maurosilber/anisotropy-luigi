from setuptools import setup

setup(
    name='anisotropy-luigi',
    version='0.0.2',
    packages=['anisotropy_luigi'],
    package_data={'': ['*.cfg']},
    url='https://github.com/maurosilber/anisotropy-luigi',
    author='Mauro Silberberg',
    author_email='maurosilber@gmail.com',
    install_requires=['donkeykong', 'tifffile', 'cellment', 'anisotropy', 'pandas', 'parse', 'numpy',
                      'scipy', 'scikit-image', 'scikit-learn']
)
