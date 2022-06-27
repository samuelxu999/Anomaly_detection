# SSA_CPD
Implementation of the SSA-based change point detection algorithm

The overview of organization of project are:

## Source code
The private swarm network configuration folder, including:

|   source   | Description |
|:----------:|-------------|
| data | sample data files for test and demo |
| utils/utilities.py | Supprt functions for ssa algorothm and application. |
| performance_analysis.py | Visualize test results for evaluation and analysis. |
| ssa.py | Core functions for Singluar Spectrum Analysis. |
| test_main.py | CLI for test and demo. |

## Usages
You can run demo cases to learn how SSA can reconstruct signals and perform change point detection.

1) To get help information, run:

	$python3 test_main.py -h


2) For SSA reconstruct data demo:

	$python3 test_main.py --test_func 0 --show_fig


3)  For SSA detection demo: 

	$python3 test_main.py --test_func 1 --op_status 2 --show_fig
