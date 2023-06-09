import sys
from copy import deepcopy
import math

#Gets the D matrix from the belief propogation formula.
#It is designed as a dictionary of two matrices. 
#One for R and one for D
def get_D(f1, f2, labels):
    
    data = {}
    for lab in labels:
        data[lab] = []
        
    with open(f1, 'r') as file:
        for line in file:
            data[labels[0]].append([ int(i) for i in line.split() ])
    
    with open(f2, 'r') as file:
        for line in file:
            data[labels[1]].append([ int(i) for i in line.split() ])
    
    return data
    

#Creates initial messages for each house in the city
#These messaage will simply be 0 everywhere as there are no meaningful mesages
#at time = 0.
#The data structe here is a triple nested dictionary.
#The outer layer is the node the message is to
#The second layer is the node the message is from
#The last layer is the labels R and D
#So the value contains in each of these locations can be seen as the value of 
#how the node "from" sees the node "to" if it chooses the specified label.
def get_m(n):

    m = {}
    #Loop through each hours in the city
    for r in range(n):
        for c in range(n):
            
            m_from = {}
            
            #Set the current value as the origin of the message
            origin = (r, c)
            
            #Check if we can send a message up
            if r > 0:
                m_from[(r-1,c)] = {'R': 0, 'D': 0}
            
            #Check if we can send a message right
            if c < (n - 1):
                m_from[(r,c+1)] = {'R': 0, 'D': 0}
                
            #Check if we can send a message down
            if r < (n - 1):
                m_from[(r+1,c)] = {'R': 0, 'D': 0}
                
            #Check if we can send a message left
            if c > 0:
                m_from[(r,c-1)] = {'R': 0, 'D': 0}

            m[origin] = m_from
            
    return m
            
            
#belief propagation algorithm given the values for D and m
#Note that we don't need a V matrix as V is just 0 if i = j
#and is 1000 otherwise.
def belief_propagation(bribes, messages, n):
    labels = ["R", "D"]
    results = [["" for j in range(n)] for i in range(n)]
    for itt in range(1000):
        new_messages = deepcopy(messages)
        
        all_messages = []
        
        for i in range(n):
            for j in range(n):
                
                #set the origin of the message and all of its neighboring houses.
                origin = (i,j)
                neighbors = [house for house in messages[(i,j)]]
                
                #Loop through all of the places that a message can go to
                for target in neighbors:
                    
                    #Looping through the distribution for the reciving node
                    for rec_lab in labels:
                        
                        #Set a temporary array for the min part of the equation
                        minimzing_array = []
                        
                        #Looping through the distribution for the sending node
                        for send_lab in labels:
                            
                            #Get the D part of the equation. (Node info)
                            D = bribes[send_lab][i][j]
                            
                            #Get the V part of the equation. (Pair info)
                            V = 0 if send_lab == rec_lab else 1000
                            
                            #Get the m parts of the equation. (messages from other neightbors)
                            M = 0
                            for node in neighbors:
                                if node != target:
                                    M += messages[origin][node][send_lab]
                            
                            minimzing_array.append(D + V + M)

                        new_messages[target][origin][rec_lab] = min(minimzing_array)
                        
                        
                        all_messages.append(min(minimzing_array))
                            
        #Prevent overflow error
        for i in range(n):
            for j in range(n):
                origin = (i,j)
                neighbors = [house for house in messages[(i,j)]]
                for node in neighbors:
                    for lab in labels:
                        new_messages[origin][node][lab] -= min(all_messages) 
                    
                    
        #Set the old messages as the new messages for the next iteration
        #Do the same for result table
        messages = new_messages      
        old_results = deepcopy(results)
        
        #Get the predicted labels given the current iteration and messages
        results = get_labels(bribes, messages, n)
        
        #Get the current cost of the results
        cost = get_cost(results, n, bribes)
        
        ##Check to see if the table is still changing
        if table_similarity(results, old_results) == 0:
            return results, cost
    
    return results, cost

#Calculate how much two tables differ
#This is used as a stopping condition if the tables in two iterations are identical       
def table_similarity(results, old):
    count  = 0
    for i in range(len(results)):
        for j in range(len(results[0])):
            if results[i][j] != old[i][j]:
                count += 1     
    return count
        

#Run through a given table and calculate the cost
def get_cost(table, n, bribes):
    pay = 0
    fences = 0
    for r in range(n):
        for c in range(n):
            
            label = table[r][c]
            
            pay += bribes[label][r][c]
            
            
            #Check if we can send a message up
            if r > 0:
                if table[r-1][c] != label:
                    fences += 1000
            
            #Check if we can send a message right
            if c < (n - 1):
                if table[r][c + 1] != label:
                    fences += 1000
                
            #Check if we can send a message down
            if r < (n - 1):
                if table[r+1][c] != label:
                    fences += 1000
                
            #Check if we can send a message left
            if c > 0:
                if table[r][c-1] != label:
                    fences += 1000

    return int(pay + (fences / 2))
    
    
#Get the classified labels given a set of messages
def get_labels(bribes, messages, n):
    labels = ["R", "D"]
    final_classes = [["" for j in range(n)] for i in range(n)]
    
    #Loop through the rows and columns of a city
    for i in range(n):
        for j in range(n):
            
            #set the origin of the message and all of its neighboring houses.
            origin = (i,j)
            neighbors = [house for house in messages[(i,j)]]
            
            #Computer the distribution of labels accross all messages
            lab_distribution = []
            for lab in labels:
                D = bribes[lab][i][j]
                M = 0
                for node in neighbors:
                    M += messages[origin][node][lab]
                lab_distribution.append(D + M)
                
            #Set the predicted value based off of which one is smaller
            if lab_distribution[0] > lab_distribution[1]:
                final_classes[i][j] = "D"
                
            else:
                final_classes[i][j] = "R"
    return final_classes

if __name__ == "__main__":

    if(len(sys.argv) != 4):
        raise(Exception("Error: expected a test case filename"))
        
    D = get_D(sys.argv[2], sys.argv[3], ["R", "D"])
    
    m = get_m(int(sys.argv[1]))
    
    print("Computing optimal labeling:")
    
    results, cost = belief_propagation(D, m, int(sys.argv[1]))
    
    
    for r in results:
        print(r)
    print("Total cost =", cost)
    
    
