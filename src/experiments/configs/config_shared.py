import argparse


def prs_parser_setup(prs: argparse.ArgumentParser):
    """Setup configuration options

    Returns:
        prs(`argparse.ArgumentParser`): The parameter arg parser that been set up

    """
    prs.add_argument(
        "--w",
        dest="weight",
        type=int,
        default=2,
        required=False,
        help="Weight coefficient",
    )
    prs.add_argument(
        "-ggi", action="store_true", default=True, help="Run GGI algorithm or not"
    )
    prs.add_argument(
        "--n_action",
        dest="n_action",
        type=int,
        default=2,
        required=False,
        help="Number of actions",
    )
    prs.add_argument(
        "--n_state",
        dest="n_state",
        type=int,
        default=3,
        required=False,
        help="Number of states",
    )
    prs.add_argument(
        "--n_group",
        dest="n_group",
        type=int,
        default=3,
        required=False,
        help="Number of groups",
    )
    return prs
