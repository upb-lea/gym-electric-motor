from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(name='gym_electric_motor',
      version='0.0.1',
      description='An OpenAI gym environment for electric motor control.',
      install_requires=requirements,
      extra_requires={'examples': ['keras>=2.2.4',
                                   'keras_rl>=0.4.2',
                                   'rl>=3.0']})
