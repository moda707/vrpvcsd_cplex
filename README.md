# Mixed Integer Programming formulation for the VRP with Highly variable Customers and Stochastic Demands
This mathematical model is presented and used in [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4251159) paper as a benchmark to compare with.

The mathematical formulation is solved with Cplex package in python. The VRPVCSD is highly complex and it cannot be solved for large instances. The presented code here solves the deterministic version of the VRPVCSD; i.e., for a realized set of customers with known demands.

To solve an example problem, please run the following command. This command solves a VRPVCSD with 10 customers and 2 vehicles, where the capacity of vehicles is 50 and the duration limit to finish tasks is 150 units of time:
'''python
python main.py 10 2 150 50

'''
