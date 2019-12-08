from setuptools import setup

setup(name='graph_conv_net',
      version='0.1',
      description='a swag MPNN',
      url='https://github.com/raph333/masters_project',
      author='Raphael Peer',
      author_email='raphael1peer@gmail.com',
      license='MIT',
      packages=['swag_net'],
      zip_safe=False,
      install_requires=['pandas', 'torch', 'numpy'])

