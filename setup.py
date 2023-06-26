from setuptools import setup, find_packages

setup(
    name = 'product_key_memory',
    packages = find_packages(),
    version = '0.2.2',
    license='MIT',
    description = 'Product Key Memory',
    long_description_content_type = 'text/markdown',
    author = 'Aran Komatsuzaki, Phil Wang',
    author_email = 'aran1234321@gmail.com, lucidrains@gmail.com',
    url = 'https://github.com/lucidrains/product-key-memory',
    keywords = ['transformers', 'artificial intelligence'],
    install_requires=[
        'einops>=0.6',
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
