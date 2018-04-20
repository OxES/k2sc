from setuptools import setup

setup(name='K2SC',
      version='1.0',
      description='K2 light curve detrending with Gaussian Processes.',
      long_description='K2 light curve detrending with Gaussian Processes.',
      author='Suzanne Aigrain',
      author_email='suzanne.aigrain@physics.ox.ac.uk',
      url='https://github.com/OxES/k2sc',
      package_dir={'k2sc':'src'},
      scripts=['bin/k2sc','bin/k2plot','bin/k2ginfo', 'bin/k2mastify'],
      packages=['k2sc'],
      install_requires=["numpy", "astropy", "scipy", "george>=0.3"],
      license='GPLv3',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Programming Language :: Python"
      ]
     )
