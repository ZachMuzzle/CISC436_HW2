import sys
import pandas as pd
import numpy as np
import re
import ast
#TAKE IN INPUT
def input_file(input):
    with open(input, 'r') as i, open(input, 'r') as j:
        lines_states = i.readlines()
        lines_observations = j.readlines()

    d_lines_s = lines_states[2::3]
    d_lines_s = list(map(lambda s: s.strip(),d_lines_s))

    d_lines_o = lines_observations[1::3]
    d_lines_o = list(map(lambda s: s.strip(),d_lines_o))
    return d_lines_o, d_lines_s

#CREATES TRANSITION MATRIX
def trans_mat(mat):
    matrix = pd.crosstab(pd.Series(mat[1:], name='Current'), pd.Series(mat[:-1], name='Next Trans'), normalize=0)

    return matrix
#CREATES EMISSION MATRIX
def emiss_mat(observations, states):
    data = pd.DataFrame([states,observations]).T
    print(data)
    data.columns = ['States','Observations']
    matrix = pd.crosstab(data.States,data.Observations,normalize=0)

    return matrix

#VITERBI ALGORITHM RETURNS ALL PREDICTED STATES
def viterbi(y, a, b):
    states = ['+', '-']
    states_dic = {'+': 0, '-': 1}
    sequence_syms = {'A': 0, 'C': 1, 'G': 2, 'T':3}
    print(a)
    # node values stored during viterbi forward algorithm
    node_values = np.zeros((len(states), len(y)))

    # probabilities of going to end state
    #end_probs = [0.1, 0.1]
    # probabilities of going from start state
    start_probs = [0.5, 0.5]

    # storing max symbol for each stage
    max_syms = np.chararray((len(states), len(y)))

    for i, sequence_val in enumerate(y):
        for j in range(len(states)):
            # if first sequence value then do this
            if (i == 0):
                node_values[j, i] = np.log(start_probs[j]) + np.log(b[j, sequence_syms[sequence_val]])
            # else perform this
            else:
                values = [node_values[k, i - 1] + np.log(b[j, sequence_syms[sequence_val]]) + np.log(a[k, j]) for k in
                          range(len(states))]

                max_idx = np.argmax(values)
                max_val = max(values)
                max_syms[j, i] = states[max_idx]
                node_values[j, i] = max_val

    # end state value
    end_state = node_values[:, -1],
    end_state_val = max(end_state)
    end_state_max_idx = np.argmax(end_state)
    end_state_sym = states[end_state_max_idx]

    # Obtaining the maximum likely states
    max_likely_states = [end_state_sym]

    prev_max = end_state_sym
    for count in range(1, len(y)):
        current_state = max_syms[:, -count][states_dic[prev_max]].decode('utf-8')
        max_likely_states.append(current_state)
        prev_max = current_state

    max_likely_states = max_likely_states[::-1]
    #[print(x) for x in max_likely_states]
    [print(max_likely_states[x]) for x in range(100)]
    return max_likely_states

#CREATES A WHOLE NEW LIST WITH A FUNCTION ORGNIZE_LIST WHICH REMOVES COMMAS,QUOTES,ETC
def new_list(items):
    counter = 0
    new_list = []

    while(counter != 100):
        #print("COUNTER is: ", counter)
        new_items = items[counter]
        #print("COUNTER is:",counter)
        items_sub = list(new_items)
        new_list.append(items_sub)

        counter += 1

    orginezed_lst = orgnize_lst(new_list)

    return orginezed_lst

#ORGANIZE THE LIST GIVEN
def orgnize_lst(lst):
    foo = str(lst)
    string_states = foo.replace('[','').replace(']','') # removes all brackets

    print("################### WHOLE NEW LIST ###################")
    li = re.split(r'[,]+\s+',string_states) # removes commas, spaces, etc. Creates new list
    print(li)
    li2 = [ast.literal_eval(x) for x in li] # removes double quotes

    return li2

#TODO: Need to add in second input(test). Also only get the model(transition matrix) and (emission matrix) for train
#      and for test use those models to use the viterbi algorithm on the test and compare the actual given value to
#      the predicted.
if __name__ == "__main__":
    file_input = sys.argv[1]
    observations, states = input_file(file_input)

   # trans_matrix = trans_mat(states)
    #print(trans_matrix)

    print("################################ OBSERVATIONS #############################")
    print(observations)
    #print(len(observations))
    print("################################ STATES ###################################")
    print(states)

    new_states = new_list(states)
    new_observations = new_list(observations)
    print(new_states)
    print(new_states[0])
    print(new_states[1])
    print(new_observations)
    print(new_observations[0])
    print(new_observations[1])
    print(new_observations[2])

    counter = 0
    counter2 = 0
    #Probilities of first 100 states
    for i in range(0,100):
        if new_states[i] == '+':
            counter += 1
        else:
            counter2 += 1
    plus_prob = counter/100
    minus_prob = counter2/100

    print(plus_prob,"\n",minus_prob,"\n")
    """
    if new_states[0] == '+':
        print("CORRECT")
    else:
        print("NOT CORRECT")

    if new_observations[0] == 'A':
        print("CORRECT")
    else:
        print("NOT CORRECT")
    """
    trans_matrix = trans_mat(new_states)
    print("#################### TRANSITION MATRIX ###############")
    print(trans_matrix)
    emission_matrix = emiss_mat(new_observations,new_states)
    emission_matrix_np = emission_matrix.to_numpy()
    print("################# EMISSION MATRIX #################")
    print(emission_matrix)
    print(emission_matrix_np)
    np_o =np.array(new_observations)
    print(new_observations[0])
    new_ob_len = len(new_observations)
    print(new_ob_len)
    # Changes observations into numbers
    print(new_observations)
    """
    for i in range(new_ob_len):
        if new_observations[i] == 'A':
            new_observations[i] = 0
        elif new_observations[i] == 'C':
            new_observations[i] = 1
        elif new_observations[i] == 'G':
            new_observations[i] = 2
        elif new_observations[i] == 'T':
            new_observations[i] = 3
    """
    new_observations = np.array(new_observations)
    trans_matrix = trans_matrix.to_numpy()
    emission_matrix = emission_matrix.to_numpy()

    x = viterbi(new_observations,trans_matrix,emission_matrix)
    for i in range(0,100):
       print(x[i])

    #pprint(x.tolist())
   # print(states[0]) # First element of list e.g +-++---
    #states_sub = states[0]
    #print(states_sub[::1]) #First element of sub-list e.g +
    #print(list(states_sub))
    #print(len(states))

