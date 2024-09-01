import argparse


def str2bool(v):
    """
    Convert string to boolean.

    Args:
        v: The input string.

    Returns:
        bool: The corresponding boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """
    Parse command line arguments and return the argument object.

    Returns:
        argparse.Namespace: An object containing all parsed arguments
    """
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Program configuration parameters")

    # Add bunchsize parameter
    parser.add_argument(
        "--bunchsize", type=int, default=64, help="Set the bunch size (default: 64)"
    )

    # Add direct parameter
    parser.add_argument(
        "--direct",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Set the direct option (default: True)",
    )

    # Add with_robust_fitting parameter
    parser.add_argument(
        "--with_robust_fitting",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Whether to use robust fitting (default: True)",
    )

    # Add aoR_cutoff parameter
    parser.add_argument(
        "--aoR_cutoff",
        type=float,
        default=1e-8,
        help="Set the aoR cutoff value (default: 1e-8)",
    )

    # Add backend parameter
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        help="Set the backend framework (default: torch)",
    )

    # Add robust_fitting_tune parameter
    parser.add_argument(
        "--robust_fitting_tune",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Whether to tune robust fitting (default: False)",
    )

    # Parse command line arguments
    args = parser.parse_args()
    return args


# When running this script directly, print all parameters
if __name__ == "__main__":
    args = get_args()
    print("Current configuration:")
    print(f"bunchsize: {args.bunchsize}")
    print(f"direct: {args.direct}")
    print(f"with_robust_fitting: {args.with_robust_fitting}")
    print(f"aoR_cutoff: {args.aoR_cutoff}")
    print(f"backend: {args.backend}")
    print(f"robust_fitting_tune: {args.robust_fitting_tune}")
