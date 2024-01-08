export POETRY_HOME=/opt/poetry
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry==1.3.2
$POETRY_HOME/bin/poetry --version