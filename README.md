# Homework Two Hidden Markov Model

## How to Compile the Program

* Within the command line run `python3 muzzleman_z_hmm.py hw2_train_labeled.data hw2_test.data `
  
## Output When Program is Ran

* A transition matrix will appear followed by an emission matrix
* The viterbi algorithm will not display because it will just print a list of states predicted. If you want to test you can place print statments after the `    x = viterbi(new_observations_test,trans_matrix,emission_matrix)` line.
* **RECALL**, **PRECISION**, **ACCURACY**, and **F1** are finally printed next
  
## Outputs of Recall, Precision, Accuracy, and F1
`RECALL: 0.26875 PRECISION: 0.6515151515151515 ACCURACY: 0.7227722772277227 F1: 0.3805309734513274`
