from setuptools import setup, find_packages
from UNDP_TASLEW import __version__

setup(name='UNDP_TASLEW',
      version=__version__,
      description='A package built by the Crisis Risk and Early Warning team to extract, transform, and analyze Twitter and other text-based data sources.',
      url='',
      author='Ethan Harrison',
      author_email='eh0097@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pandas',
          'sklearn',
          'gensim',
          'corextopic',
          'huggingface',
          'nltk',
        #   'collections',
          'future',
        #   'bulletins',
          'pandas_flavor',
          'corextopic',
          'detoxify'],
        
        )


