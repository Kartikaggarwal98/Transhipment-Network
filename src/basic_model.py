
import numpy as np
import gurobipy as gbp
import time
np.random.seed(352)


Si=np.array([[100],[200]])
Di=np.array([[75],[150],[200]])

# Indices & Variable Names
supply_nodes = 2
demand_nodes = 3
supply_nodes_range = range(supply_nodes)
demand_nodes_range = range(demand_nodes)
all_nodes_len = supply_nodes*demand_nodes
ALL_nodes_range = range(all_nodes_len)


# Create Model, Set MIP Focus, Add Variables, & Update Model
m = gbp.Model(' -- The Transportation Problem -- ')
# Set MIP Focus to 2 for optimality
m.setParam('MIPFocus', 1)
m.setParam(gbp.GRB.Param.PoolSearchMode, 1)
m.setParam(gbp.GRB.Param.PoolGap, 0.10)
decision_var = []
for orig in supply_nodes_range:
    decision_var.append([])
    for dest in demand_nodes_range:
        decision_var[orig].append(m.addVar(vtype=gbp.GRB.INTEGER, 
#                                         obj=Cij[orig][dest],
#                                            obj=1,
                                        name='S'+str(orig+1)+'_D'+str(dest+1)))
# Update Model Variables
m.update()       

m.setObjective(gbp.quicksum(int(Di[dest])-gbp.quicksum(decision_var[orig][dest] for orig in supply_nodes_range)
                            for dest in demand_nodes_range), 
                        gbp.GRB.MINIMIZE)

m.update()

m.display()

# Add Supply Constraints
for orig in supply_nodes_range:
    m.addConstr(gbp.quicksum(decision_var[orig][dest] 
                        for dest in demand_nodes_range) - Si[orig] <= 0)
# Add Demand Constraints
for orig in demand_nodes_range:  
    m.addConstr(gbp.quicksum(decision_var[dest][orig] 
                        for dest in supply_nodes_range) - Di[orig] <= 0)

#  Optimize and Print( Results)
m.optimize()
m.write('path.lp')

nSolutions = m.SolCount
print (nSolutions)
for e in range(nSolutions):
        m.setParam(gbp.GRB.Param.SolutionNumber, e)
        print('%g ' % m.PoolObjVal, end='')
        print ([(v.Varname,v.x) for v in m.getVars()])
