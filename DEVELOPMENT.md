# Packaging
Update build tool: `python -m pip install --upgrade build`

Build: `python -m build`

# Testing 
Run: `pytest --sw`
> --sw, --stepwise      Exit on test failure and continue from last failing test next time

Warning as error:
> python -W error -m pytest --sw -v

# Linter and Formater
Ruff: https://docs.astral.sh/ruff/installation/

Use `ruff check src/` for linting
or `ruff format src/` for formatting


# Install package for local development
Run: `pip install -e .`
> -e, --editable <path/url>   Install a project in editable mode (i.e. setuptools "develop mode") from a local project path or a VCS url

Check correct package install directory with python interpreter
Run: `python`

```
>>> import gym_electric_motor as gem
>>> gem
<module 'gym_electric_motor' from '/home/***/gym-electric-motor/src/gym_electric_motor/__init__.py'>
```

# Sidenotes
```
python -V
Python 3.10.13
```


## No poetry
Some complex dependency systems don't work good with poetry (e.g. pytorch) (https://python-poetry.org/)