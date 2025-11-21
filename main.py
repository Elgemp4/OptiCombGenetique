from email.policy import default

import click

from genetic import genetic
from parser import read_file


@click.command()
@click.argument("file")
def enter_point(file):

    genetic(X, r, LW, UW, LH, UH)