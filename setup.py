from setuptools import setup

setup(
    name='anisotropy-luigi',
    version='0.0.1',
    packages=['anisotropy_luigi'],
    url='https://github.com/maurosilber/anisotropy-luigi',
    author='Mauro Silberberg',
    author_email='maurosilber@gmail.com',
    install_requires=['luigi', 'cellment', 'anisotropy', 'pandas', 'parse', 'numpy', 'scipy', 'scikit-image']
)
