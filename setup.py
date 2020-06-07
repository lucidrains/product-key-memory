from setuptools import setup, find_packages

setup(
  name = 'product_key_memory',
  packages = find_packages(),
  version = '0.0.7',
  license='MIT',
  description = 'Product Key Memory',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/product-key-memory',
  keywords = ['transformers', 'artificial intelligence'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)