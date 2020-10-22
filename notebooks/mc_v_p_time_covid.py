#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gurobipy as gbp
import time
import pandas as pd
np.random.seed(352)


# # Commodity
# 
# ppe: 80 (36*2 + 8)
# ventilators: 1*0.7/person/day
# oxy: 4/person/day 
# hcq: 3/p/d
# paracetamol: 5/p/d
# syringes: 68/p/d
# infusion_pump: 2.5/p/d
# 
# Delhi cases: 3600 per day(critical). 
# 
# Total pateint -> 72,000
# 5% ICU =3600 , 15% ACU, 80% SCU
# 
# 1. 70% of ICU require ventilator -> 3600*70% = 2520
# 2. Infusion pump -> 9000
# 3. Sanitizer -> 3600
# 4. HCQ -> 10,800
# 5. Oxygen -> 14,400*


n_com=4 #no of commodities
n_sup=2 #no of supply nodes
n_dem=3 #no of demands nodes

#Supply Demand of all commodities 
"""[
[Day1:[Supplier_1:comm1,Supplier_1:comm2],[s2c1,s2c2]],
Day2:[],
Day3:[]
    ]"""

Si=np.array([[[700,2500,7000,5000],[400,2500,3000,4000]],
             [[1000,6000,10000,11000],[500,4000,6000,4000]],
             [[1500,6000,30000,13000],[1000,4000,10000,7000]]])
Dj=np.array([[[800,1100,5000,4000],[500,700,3000,2000],[1200,1800,7000,5000]],
             [[1200,1500, 6000,5000],[800,500,5000,5000],[2000,4000,12000,8000]],
             [[2000,4000, 10000, 6000],[1000,10000, 6000, 4000],[4000,4000,20000, 17000]]])
Si.shape,Dj.shape #days X nodes X commodities

# check assumption that demand is always higher than supply
np.sum(Si,axis=1)<np.sum(Dj,axis=1)


#Vehicle capacity, number of vehicles with each supplier, cost of transportation from supply nodes to demand nodes
V_cap=3000
V_num=np.array([15,20])
V_cost=np.array([[10,15,20],[18,13,20]])*100

#People Variables
n_pep=np.array([[[1000,1000,1000,1000],[800,800,800,800],[1800,1800,1800,1800]],
                [[2200,2200,2200,2200],[1100,1100,1100,1100],[2400,2400,2400,2400]],
                [[3000,3000,3000,3000],[1900,1900,1900,1900],[4100,4100,4100,4100]]])
# n_pep=np.array([[[25,25],[50,50],[30,30]],
#                 [[25,25],[50,50],[30,30]],
#                 [[25,25],[50,50],[30,30]]]) #no of people on each demand node,
mu=[0.7,1,4,3] #consumption rate (comm req per person per day)
pc=[0.8,0.2,0.5,1.0]

n_days=len(Si)
days_range=range(n_days)
days_range

# Indices & Variable Names
supply_nodes = n_sup
demand_nodes = n_dem
supply_nodes_range = range(n_sup)
demand_nodes_range = range(n_dem)
comm_range=range(n_com)
all_nodes_len = n_sup*n_dem
ALL_nodes_range = range(all_nodes_len)

print (supply_nodes_range, demand_nodes_range,comm_range, all_nodes_len)


Pc=np.array([Dj[ix]/Dj[ix].sum(axis=0) for ix in range(len(Dj))])
Pc=Pc*pc

dep_time=24
dep_cost=np.exp(1.5031+0.1172*dep_time)-np.exp(1.5031)
dep_cost


unserved_people=[[[0,0,0,0],[0,0,0,0],[0,0,0,0]]]
prev_inventory=[Si[0]*0]
unserved_people,prev_inventory
cols=['Day','S_Node']+['D'+str(i+1)+'_c'+str(j+1) for i in demand_nodes_range for j in comm_range]
cols+=['Total_S_c'+str(i+1) for i in comm_range]
cols+=['Deliver_S_c'+str(i+1) for i in comm_range]
cols+=['D'+str(i+1)+'_v' for i in demand_nodes_range]
cols+=['Total_V']

df=pd.DataFrame(columns=cols,index=None,dtype=int)

cols2=['Day','D_Node']+['Total'+'_c'+str(i+1) for i in comm_range]+['Unserv'+'_c'+str(i+1) for i in comm_range]
df2=pd.DataFrame(columns=cols2,index=None,dtype=int)


# unserved_people=[[[0,0],[0,0],[0,0]]]
for day in days_range:
    print ('#'*50,'  Day',day+1,'  ','#'*50)
    # Create Model, Set MIP Focus, Add Variables, & Update Model
    m = gbp.Model(' -- The Multi Commodity Vehicle Transportation Problem -- ')

    # Set MIP Focus to 2 for optimality
    m.setParam('MIPFocus', 2)
    # m.setParam(gbp.GRB.Param.PoolSearchMode, 1)
    # m.setParam(gbp.GRB.Param.PoolGap, 0.10)

    decision_var = []
    vehicles_var=[]
    unserved_var=[]
    for orig in supply_nodes_range:
        decision_var.append([])
        vehicles_var.append([])
        for dest in demand_nodes_range:
            decision_var[orig].append([])
            vehicles_var[orig].append(m.addVar(vtype=gbp.GRB.INTEGER,
                                              name='S'+str(orig+1)+'_D'+str(dest+1)+'_V'))
            for comm in comm_range:
    #             print (comm,decision_var)
                decision_var[orig][dest].append(m.addVar(vtype=gbp.GRB.INTEGER, 
    #                                         obj=Cij[orig][dest],
    #                                            obj=1,
                                            name='S'+str(orig+1)+'_D'+str(dest+1)+'_c'+str(comm+1)))
    for dest in demand_nodes_range:
        unserved_var.append([])
        for comm in comm_range:
            unserved_var[dest].append(m.addVar(vtype=gbp.GRB.INTEGER,
                                              name='D'+str(dest+1)+'_c'+str(comm+1)+'_U'))
            
########     Update Model Variables
    m.update() 
    
    #sum(sum[(Demand - net supplied)*priority for every demand node] for every commodity)
    first_term=10*gbp.quicksum(gbp.quicksum((int(Dj[day][dest][comm])-gbp.quicksum(decision_var[orig][dest][comm] for orig in supply_nodes_range))*(Pc[day][dest][comm])
                                for dest in demand_nodes_range) for comm in comm_range)
    print ('First term: ',first_term)
    
    second_term=0.01*gbp.quicksum(gbp.quicksum(vehicles_var[orig][dest]*V_cost[orig][dest] for dest in demand_nodes_range) for orig in supply_nodes_range)
    third_term=0.1*gbp.quicksum(gbp.quicksum(unserved_var[dest][comm]*dep_cost*(1/pc[comm]) for comm in comm_range) for dest in demand_nodes_range)
#     third_term=0.1*gbp.quicksum(gbp.quicksum(unserved_var[dest][comm] for comm in comm_range) for dest in demand_nodes_range)
    
    #objective function
    m.setObjective(first_term+second_term+third_term,gbp.GRB.MINIMIZE)

    m.update()

##########     Add Supply Constraints

    #sum(net supplied) <= available supply + previous inventory
    for orig in supply_nodes_range:
        for comm in comm_range:
            m.addConstr(gbp.quicksum(decision_var[orig][dest][comm]
                                     for dest in demand_nodes_range) - Si[day][orig][comm] - prev_inventory[day][orig][comm] <= 0)
            
########     Add Demand Constraints
    #sum(supplied commodity at demand node) <= demand that day
    for dest in demand_nodes_range:  
        for comm in comm_range:
            m.addConstr(gbp.quicksum(decision_var[orig][dest][comm] 
                                     for orig in supply_nodes_range) - Dj[day][dest][comm] <= 0)
            
###########     Add vehicle constraints

#     for orig in supply_nodes_range:
#         m.addConstr(gbp.quicksum(decision_var[orig][dest][comm]
#                                  for dest in demand_nodes_range for comm in comm_range) - V_cap*V_num[orig] <=0)

    #total sent vehicles at demand nodes <= available vehicles
    for orig in supply_nodes_range:
        m.addConstr(gbp.quicksum(vehicles_var[orig][dest] for dest in demand_nodes_range) - V_num[orig] <=0)
    
    #vehicles sent to a demand node >= (total supplied items to that node / vehicle capacity) 
    for orig in supply_nodes_range:
        for dest in demand_nodes_range:
            m.addConstr(-sum(decision_var[orig][dest][comm]
                                for comm in comm_range)/V_cap + vehicles_var[orig][dest]>=0)
            
    # vehicles sent to a demand node-1 <= (total supplied items to that node / vehicle capacity)
    for orig in supply_nodes_range:
        for dest in demand_nodes_range:
            m.addConstr(-sum(decision_var[orig][dest][comm]
                                for comm in comm_range)/V_cap + vehicles_var[orig][dest]<=1)

            
######     Add unserved people contstraints
    #unserved people at t <= num of people at t + unserved people at t-1
    for dest in demand_nodes_range:
        for comm in comm_range:
#             print (dest,comm,day,np.array(unserved_var).shape,np.array(n_pep).shape,np.array(unserved_people).shape)
#             print (unserved_var[dest][comm])
#             print (n_pep[day][dest][comm])
#             print (unserved_people[day][dest][comm])
            m.addConstr(unserved_var[dest][comm]<=n_pep[day][dest][comm]+unserved_people[day][dest][comm])
            
    # supplied commodity at demand node > (no of people at t + unserved people at t-1 - unserved people at t)*consumption rate
    for dest in demand_nodes_range:
        for comm in comm_range:
            m.addConstr(sum(decision_var[orig][dest][comm] for orig in supply_nodes_range)-
                        ((n_pep[day][dest][comm]+unserved_people[day][dest][comm]-unserved_var[dest][comm])*mu[comm])>=0)
            
        
#      Adding 0 inventory for next day constraint: no supply left for t+1
#     for orig in supply_nodes_range:
#         for comm in comm_range:
#             m.addConstr(gbp.quicksum(decision_var[orig][dest][comm]
#                                      for dest in demand_nodes_range) - Si[day][orig][comm] >= 0)

    #  Optimize and Print( Results)
    m.optimize()
    m.write('./path.lp')
#     print (m.display())

#     m.display()
    prev_inventory.append([])

    #todays supply + previous inventory - net supplied = todays inventory
    for orig in supply_nodes_range:
        prev_inventory[day+1].append([])
        for comm in comm_range:
            prev_inventory[day+1][orig].append(Si[day][orig][comm]+prev_inventory[day][orig][comm]-sum(decision_var[orig][dest][comm].x
                                 for dest in demand_nodes_range))

#     print (prev_inventory)
    unserved_people.append([])
    for dest in demand_nodes_range:
        unserved_people[day+1].append([])
        for comm in comm_range:
            unserved_people[day+1][dest].append(unserved_var[dest][comm].x)

            
    
    #Populate DataFrame
    for ix in supply_nodes_range:
        r_t=[]
        r_t+=[decision_var[ix][iy][iz].x for iy in demand_nodes_range for iz in comm_range]
        r_t+=[Si[day][ix][iy] for iy in comm_range]
        r_t+=[sum(decision_var[ix][iy][iz].x for iy in demand_nodes_range) for iz in comm_range]
        r_t+=[vehicles_var[ix][iy].x for iy in demand_nodes_range]
        r_t+=[V_num[ix]]
        df.loc[len(df)]=[day+1,ix+1]+r_t

    for ix in demand_nodes_range:
        r_t=[]
        r_t+=[n_pep[day][ix][iy]+unserved_people[day][ix][iy] for iy in comm_range]
        r_t+=[unserved_var[ix][iy].x for iy in comm_range]
        df2.loc[len(df2)]=[day+1,int(ix+1)]+r_t
    
    first_term_val=sum(sum((int(Dj[day][dest][comm])-sum(decision_var[orig][dest][comm].x for orig in supply_nodes_range))*(Pc[day][dest][comm])
                                for dest in demand_nodes_range) for comm in comm_range)
    second_term_val=0.01*sum(sum(vehicles_var[orig][dest].x*V_cost[orig][dest] for dest in demand_nodes_range) for orig in supply_nodes_range)
    third_term_val=0.1*sum(sum(unserved_var[dest][comm].x*dep_cost for comm in comm_range) for dest in demand_nodes_range)
    
    print ('^'*100)
    print ("First term: ",int(first_term_val)," Second term: ",int(second_term_val)," Third term: ",int(third_term_val))
    print ('^'*100)


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


x1=np.array([250,1450,1950,5,534,2534])
x2=np.array([250,950,1450,5,700,2700])

from matplotlib import pyplot as plt


x1=df2['Total_c1'].values-df2['Unserv_c1'].values
x2=df2['Total_c2'].values-df2['Unserv_c2'].values
x3=df2['Total_c3'].values-df2['Unserv_c3'].values
x4=df2['Total_c4'].values-df2['Unserv_c4'].values


barWidth=0.65
br1 = np.arange(len(x1[::2])) 
br2 = [x + barWidth for x in br1] 

br11 = [x + barWidth/4 for x in br1] 
br111 = [x + barWidth/4 for x in br11] 

br22 = [x + barWidth/4 for x in br2] 
br222 = [x + barWidth/4 for x in br22] 
# br3 = [x + barWidth for x in br2] 
# br4 = [x + barWidth for x in br3] 


fig = plt.subplots(figsize =(6, 4)) 
plt.bar(br1,[250,950,1450], color ='red',  
        width = barWidth/5,label='Oxygen with deprivation') 
plt.bar(br11,[250,1450,1950], color ='dodgerblue',  
        width = barWidth/5,label='Oxygen without deprivation') 
plt.legend(loc='best')
plt.xlabel('Week', fontweight ='bold') 
plt.ylabel('UnServed Patients', fontweight ='bold') 
# plt.xticks([r + barWidth for r in range(len(x1[::3]))], 
#            ['1', '2', '3']) 
plt.savefig('UnServedPatOxy.png',dpi=400)
plt.show()

fig = plt.subplots(figsize =(6, 4)) 
plt.bar(br1,[40,700,2700], color ='red',  
        width = barWidth/5,label='HCQ with deprivation') 
plt.bar(br11,[40,534,2534], color ='dodgerblue',  
        width = barWidth/5,label='HCQ without deprivation') 
plt.legend(loc='best')
plt.xlabel('Week', fontweight ='bold') 
plt.ylabel('UnServed Patients', fontweight ='bold') 
plt.savefig('UnServedPatHCQ.png',dpi=400)


fig = plt.subplots(figsize =(6, 4)) 
plt.bar(br1,x1[::3], color ='red',  
        width = barWidth/5,label='Oxygen week-1') 
plt.bar(br2,x2[::3], color ='dodgerblue',  
        width = barWidth/5,label='HCQ week-1') 
plt.bar(br11,x1[1::3], color ='orange',  
        width = barWidth/5,label='Oxygen week-2') 
plt.bar(br22,x2[1::3], color ='b',  
        width = barWidth/5,label='HCQ week-2') 
plt.bar(br111,x1[2::3], color ='maroon',  
        width = barWidth/5,label='Oxygen week-3') 
plt.bar(br222,x2[2::3], color ='darkviolet',  
        width = barWidth/5,label='HCQ week-3')
# plt.bar(br3,np.sum(x3.reshape(-1,3),axis=1), color ='red',  
#         width = barWidth,label='Oxygen') 
# plt.bar(br4,np.sum(x4.reshape(-1,3),axis=1), color ='orange',  
#         width = barWidth,label='HCQ') 
# Adding Xticks  

plt.legend(loc='best')
plt.xlabel('Week', fontweight ='bold') 
plt.ylabel('UnServed Patients', fontweight ='bold') 
# plt.xticks([r + barWidth for r in range(len(x1[::3]))], 
#            ['1', '2', '3']) 
plt.savefig('ServedPat.png',dpi=400)
plt.show()


fig = plt.subplots(figsize =(6, 4)) 
plt.bar(br1,np.sum(x1.reshape(-1,3),axis=1), color ='green',  
        width = barWidth,label='Ventilator') 
plt.bar(br2,np.sum(x2.reshape(-1,3),axis=1), color ='blue',  
        width = barWidth,label='Sanitizer') 
plt.bar(br3,np.sum(x3.reshape(-1,3),axis=1), color ='red',  
        width = barWidth,label='Oxygen') 
plt.bar(br4,np.sum(x4.reshape(-1,3),axis=1), color ='orange',  
        width = barWidth,label='HCQ') 
# Adding Xticks  

plt.legend(loc='best')
plt.xlabel('Week', fontweight ='bold') 
plt.ylabel('Served Patients', fontweight ='bold') 
plt.xticks([r + barWidth for r in range(len(x1[::3]))], 
           ['1', '2', '3']) 
plt.savefig('ServedPat.png',dpi=400)
plt.show()


c=['g','r','b','orange']
com=['Ventilator','Sanitizer','Oxygen','HCQ']
for ix in range(4):
    plt.plot(np.sum(Dj,axis=1)[:,ix],label=com[ix],color=c[ix])
    plt.plot(np.sum(Si,axis=1)[:,ix],label=com[ix],linestyle='dashed',color=c[ix])
    
plt.legend()
plt.savefig('DemvsSup.png',dpi=400)

# for ix in range(4):
#     plt.plot(Dj[:,1,ix],color='green')
# for ix in range(4):
#     plt.plot(Dj[:,2,ix],color='b')





