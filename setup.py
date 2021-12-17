from setuptools import setup

setup(
    name='nionswift_frwr_plugin',

    version='0.1',

    description='A plugin for nionswift allowing setting up defocus values for a defocus series used in full resolution wave reconstruction (inline electron holography).',
    long_description='',

    author='Griogry Kornilov',
    author_email='kornilog@hu-berlin.de',

    license='GNU General Public License v3.0',
    url='https://git.physik.hu-berlin.de/kornilog/nionswift_frwr_plugin',
    download_url = 'https://git.physik.hu-berlin.de/kornilog/nionswift_frwr_plugin/-/archive/main/nionswift_frwr_plugin-main.tar.gz',
    keywords = ['NIONSWIFT', 'FRWR', 'DEFOCUS', 'PLUGIN'],
    packages=['nionswift_plugin.nionswift_frwr_plugin'],
    install_requires=['h5py', 'nionutils', 'nionui', 'nionswift'],
    package_data={'':['defaults.dat']},
    include_package_data=True,
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    ],
   )