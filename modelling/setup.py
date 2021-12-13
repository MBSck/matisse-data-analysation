from setuptools import setup
setup(name='modelling',
      version='0.1.1',
      description='A package for visibility modelling',
      url='#',
      author='Marten Scheuck',
      author_email='martenscheuck@gmail.com',
      license='LICENSE.txt',
      packages=['modelling', 'modelling.functionality',
               'modelling.models', 'modelling.assets'],
      long_description=open('README.txt').read(),
      install_requires=[
          "pytest",
          "numpy",
          "scipy",
      ],
      zip_safe=False
)
