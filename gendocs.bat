jupyter-book build docs/
pdoc -d numpy -o docs/_build/html/api ./eckity "!eckity.*.tests"