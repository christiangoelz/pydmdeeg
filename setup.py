from setuptools import setup

setup(
    name='pydmdeeg',
    version='0.1.0',    
    description='Dynamic mode decomposition applied to EEG data',
    url='',
    author='Christian Goelz',
    author_email='c.goelz@gmx.de',
    license='MIT',
    packages=['dmdeeg'],
    install_requires=[
        'optht',
 	'pandas',
        'seaborn',
        'numpy',
        'matplotlib',
        'scipy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

