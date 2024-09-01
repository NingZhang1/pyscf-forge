import argparse


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
        "--bunchsize", type=int, default=128, help="Set the bunch size (default: 128)"
    )

    # Add direct parameter
    parser.add_argument(
        "--direct",
        type=bool,
        default=True,
        help="Set the direct option (default: True)",
    )

    # Add with_robust_fitting parameter
    parser.add_argument(
        "--with_robust_fitting",
        type=bool,
        default=True,
        help="Whether to use robust fitting (default: True)",
    )
    
    # Add with_robust_fitting parameter
    parser.add_argument(
        "--robust_fitting_tune",
        type=bool,
        default=False,
        help="Whether to use robust fitting to tune the result from without r.t. (default: True)",
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
