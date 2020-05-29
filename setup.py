import setuptools


AUTHORS = [
    'Arne Traue', 'Gerrit Book', 'Praneeth Balakrishna',
    'Max Schenke', 'Wilhelm Kirchgässner', 'Oliver Wallscheid',
]

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='gym_electric_motor',
      version='0.2.1',
      description='An OpenAI gym environment for electric motor control.',
      packages=setuptools.find_packages(),
      install_requires=requirements,
      extras_require={'examples': 
                      ['keras_rl2>=1.0.3']
                     },
      author=', '.join(sorted(AUTHORS, key=lambda n: n.split()[-1].lower())),
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/upb-lea/gym-electric-motor",
      )
