import sys
from random import randint
import random
from random import sample 
import numpy as np
from operator import itemgetter
import pandas as pd

#Reads in csv files
def open_file(file):
    df = pd.read_csv(file, sep=',|;', engine='python')
    classifier_names = df.columns.tolist()
    return(df, classifier_names)

# Calculates fitness of the individuum
def fitness(individuum, data_type):
    # all four possible results are initialized to zero
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    # for each ID the result of classifier is calculated and compared to annotation
    for i in range(len(data_type)):    
        real_status = int(data_type[i][1])
        
        # Creates a list of SC's outputs for sample i
        classifier_results = [[]] * len(individuum)
        for j in range(len(individuum)):
            miRNA = individuum[j][1]
            if (individuum[j][0] == 0):
                classifier_results[j] = int(not int(data_type[i][miRNA]))
            else:
                classifier_results[j] = int(data_type[i][miRNA])
        #SC's outputs are evaluated using the majority vote
        if sum(classifier_results) > int(len(classifier_results)/2):
            classifier_result = 1
        else:
            classifier_result = 0
        #  Annotation is compared to classifier output  
        if (real_status == classifier_result):
            if (real_status == 1):
                true_positive += 1
            else: 
                true_negative += 1
        else:
            if (real_status == 1):
                false_negative += 1
            else: 
                false_positive += 1
    #Accuracy is used to define fitness of an individuum            
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
    return(accuracy)  

# Creates of list of individuals selected for a tournament and their fitnesses 
def calculate_tournament_fits(ids, fit_list):
    tournament_fits = [[]] * len(ids)
    for i in range(len(ids)):
        tournament_fits[i] = fit_list[ids[i]]
    return(tournament_fits)
 
# Calculates crossover for two parents    
def calculate_crossover(parent1, parent2):
    child1 = []
    child2 = []
    child1.append(parent1[0])
    child2.append(parent2[0])
    
    if (len(parent2) > len(parent1)): # making sure that len(parent1)>=len(parent2)
        temp = parent1
        parent1 = parent2
        parent2 = temp
    for i in range(1, len(parent2)): 
        crossover_temp(parent1[i], child1, child2) #randomly assigning Single Classifiers from the first parent to either of the children
        crossover_temp(parent2[i], child1, child2) #randomly assigning Single Classifiers from the second parent to either of the children
    for i in range(len(parent2),len(parent1)): #randomly assigning Single Classifiers from the longer parent to either of the children
        crossover_temp(parent1[i], child1, child2)
    return(child1, child2)

# Help function for a crossover calculation: assigning a SC of a parent to one of the children
def crossover_temp(parent, child1, child2):
    child1_miRNAs = [child1[x][1] for x in range(len(child1))] 
    child2_miRNAs = [child2[x][1] for x in range(len(child2))]
    if (parent[1] in child1_miRNAs):    #if SC is already included in a child, then it is assigned to another child if it has less than 10 SCs
        if (len(child2) < 10):
            child2.append(parent)
    elif (parent[1] in child2_miRNAs):
        if (len(child1) < 10):
            child1.append(parent)
    elif (len(child1) >= 10):  # if a child already has max amount of classifiers, then a parent classifier is assigned to another child
        child2.append(parent)
    elif (len(child2) >= 10):
        child1.append(parent)
    else: #if a parent SC can be assigned to either of the children, then it is randomly assigned
        assignment=randint(0,1)
        if (assignment == 0): 
            child1.append(parent)
        else:
            child2.append(parent)
    return
 
# Calculates a random mutation of an individuum
def mutation(individuum):
    random_mutation_id = randint(2, len(classifier_names)-1) # generates a random mutation
    ind_miRNAs = [individuum[x][1] for x in range(len(individuum))]
    
    if (random_mutation_id in ind_miRNAs):  # if the mutation to be introduced is already one of SCs of an individuum, another random mutation SC is generated
        mutated_individuum = mutation(individuum)
    else:
        random_mutation = randint(0, len(individuum)-1) #selecting a SC in an individuum to be mutated
        random_bool = randint(0,1) #randomly selecting if a mutated SC will be negated or non negated
        individuum[random_mutation] = (random_bool, random_mutation_id) # saving the mutation to the individuum
        mutated_individuum = individuum
    return(mutated_individuum)
        
def genetic_algorithm(iterations, population_size,crossover_prob, mutation_prob, tournament_size):
    temp_result = [[]] * 2
    #creating a population of the given size
    population = [[]] * population_size
    
    #randomly generating individuums in population
    for j in range(population_size):
        miRNAs = random.sample(range(2, len(classifier_names)-1), randint(1,10)) #each individuum consists of 1-10 SCs randomly chosen from all available SCs
        booleans = [randint(0,1) for x in range(len(miRNAs))] #each SC is either negated (0) or not (1)
        population[j] = [[]]*len(miRNAs)
        
        #saving the generated SCs for each individuum
        for k in range(len(population[j])):
            population[j][k] = booleans[k], miRNAs[k]
    
    # running the algorithm for the given amount of iterations
    for l in range(iterations):
        fit_list = [[]] * population_size
        
        #calculating fitness of the population
        for m in range(population_size): 
            fit_list[m] = fitness(population[m], train_data)
        
        #selecting individuums for crossover
        crossover_pool = [[]] * population_size
        for n in range(population_size): # running tournament Population_size times, so that the crossover pool has the same size as population
            tournament_ids = random.sample(range(0, population_size-1), tournament_size) #randomly selecting individen, which will participate in a tournament
            tournament_fitnesses = calculate_tournament_fits(tournament_ids, fit_list) #collecting fitnesses of the selected individuums
            selected_id = tournament_ids[np.argmax(tournament_fitnesses)] #selecting an individuum with the best fitness for the crossover pool
            crossover_pool[n] = population[selected_id] #saving the selected individuum into a crossover pool
        
        #Crossover
        new_population = [[]] * population_size
        
        for p in range(population_size//2):
            parent_ids = random.sample(range(0, population_size-1), 2) #randomly selecting 2 parents from the  crossover pool
            parent_1 = crossover_pool[parent_ids[0]]
            parent_2 = crossover_pool[parent_ids[1]]
            
            #if randomly generated number is smaller than crossover probability, then crossover takes place, and both children are saved to the next generation, otherwise parents are saved into the next generation
            random_crossover = random.uniform(0, 1)
            if (random_crossover <= crossover_prob):
                new_population[p * 2] = calculate_crossover(parent_1, parent_2)[0]
                new_population[p * 2 + 1] = calculate_crossover(parent_1, parent_2)[1]
            else:
                new_population[p * 2] = parent_1
                new_population[p * 2 + 1] = parent_2
        
        #Mutation: if a random number is smaller or equal than mutation probability, then mutation takes place
        
        for r in range(population_size):
            random_mutation = random.uniform(0, 1)
            if (random_mutation <= mutation_prob):
                new_population[r] = mutation(new_population[r])
        # at the end of an iteration new population is saved as a start population for the next iteration        
        population = new_population
    # after the last iteration the individuum with maximal fitness is returned. In case of multiple individuums with same fitness the shortest is returned   
    resulted_fitness = [fitness(population[x], train_data) for x in range(population_size)]
    indices = [i for i, x in enumerate(resulted_fitness) if x == max(resulted_fitness)]
    lengthes = [[]] * len(indices)
    for x in range(len(indices)):
        lengthes[x] = len(population[indices[x]])
    minimal_index = indices[np.argmin(lengthes)]
    temp_result[0] = population[minimal_index]
    temp_result[1] = resulted_fitness[minimal_index]
    return(temp_result)
        
        
################################################################################################################   

#Reading in files for training and testing
train_input = input("Please provide the name of the train data file: ")
test_input = input("Please provide the name of the test data file: ")

train_result = open_file(train_input)
train_df = train_result[0]

classifier_names = train_result[1]

test_result = open_file(test_input)
classifier_names_test = test_result[1]

#removing classifiers from test data, which are not in the train data
extra_columns = np.setdiff1d(classifier_names_test,classifier_names)
classifier_names_test = list(np.setdiff1d(classifier_names_test,extra_columns))

test_df = test_result[0]
test_df = test_df.drop(extra_columns, 1)

if classifier_names_test != classifier_names:
    classifier_names.remove('Annots')
    classifier_names.remove('ID')
    classifier_names.sort()
    classifier_names = ['ID', 'Annots'] + classifier_names

    classifier_names_test.remove('Annots')
    classifier_names_test.remove('ID')
    classifier_names_test.sort()
    classifier_names_test = ['ID', 'Annots'] + classifier_names_test
    

train_df = train_df[classifier_names]
test_df = test_df[classifier_names_test]  

train_data = train_df.values.tolist()
test_data = test_df.values.tolist()


#Generating 100 random parameter sets
profile = [[]] * 100
for i in range(100):
    profile[i] =  [[]] * 5
for i in range(100):
    profile[i][0] = randint(1,4) * 25  # number of iterations: random number from 25 to 100 with step 25
    profile[i][1] = randint(1,6) * 50  # population size: random number from 50 to 300 with step 50
    profile[i][2] = randint(1,10) / 10  # crossover probability: random number from 0.1 to 1 with step 0.1
    profile[i][3] = randint(1,10) / 10  # mutation probability: random number from 0.1 to 1 with step 0.1
    profile[i][4] = randint(1,5) * 10  # tournament size in % from total population: random number from 10 to 50 with step 10

results = [[]] * 100

for i in range(100): # running the algorithm for all 100 parameter sets 
    results[i] = [[]] * 5 
    # saving all parameters as local variables
    print("Iteration: ", i)
    iterations = profile[i][0]
    population_size = profile[i][1]
    crossover_prob = profile[i][2]
    mutation_prob = profile[i][3]
    tournament_size = int(profile[i][4] / 100 * population_size)
    #algorithm runs 10 times for each parameter set, results are saved to temp results
    temp_results = [[]] * 10
    for j in range(10):
        temp_results[j] = genetic_algorithm(iterations, population_size,crossover_prob, mutation_prob, tournament_size)
    
    #the best result of ten runs of the algorithm is the shortest individuum among individuums with max fitness
    results_fitnesses = [temp_results[x][1] for x in range(10)]
    indices = [i for i, x in enumerate(results_fitnesses) if x == max(results_fitnesses)]
    lengthes = [[]] * len(indices)
    for x in range(len(indices)):
        lengthes[x] = len(temp_results[indices[x]][0])
    minimal_index = indices[np.argmin(lengthes)]
    
    #Saving the results of the parameter run
    results[i][0] = profile[i] #used parameter set
    results[i][1] = temp_results[minimal_index][0] #best individuum 
    results[i][2] = results_fitnesses[minimal_index] # its fitness
    results[i][3] = np.mean(results_fitnesses) #average fitness for these paremeter set
    results[i][4] = np.std(results_fitnesses) #standard deviation
    

    
# Printing the results
results_fitn = [results[x][2] for x in range(len(results))]
id_best = np.argmax(results_fitn) 
best_classifier = results[id_best][1] # selecting the best fitting result

#replacing list of number with list of miRNAs for all classifiers
for i in range(len(results)):
    results[i][1] = sorted(results[i][1], key=itemgetter(1))
    temp = ""
    classifier = results[i][1]
    for j in range(len(classifier)):
        temp_sc = classifier[j][1]
        if classifier[j][0] == 0:
            temp += 'NOT '
        temp += str(classifier_names[temp_sc])
        temp += ', '
    results[i][1] = temp

#converting the results to dataframe and printing    
df = pd.DataFrame(results, columns =['Parameter set', 'Individuum', 'Fitness', 'Average fitness', 'Standard deviation'])   
pd.set_option('max_columns', None)
pd.set_option("max_rows", None)

print(df)

#printing all the information for the best result and checking its fitness for the test data
best_result = results[id_best]
print("BEST RESULT:", '\n', "Number of iterations: ", best_result[0][0], '\n', "Population size: ", best_result[0][1], '\n', "Crossover probability: ", best_result[0][2], '\n', "Mutation probability: ", best_result[0][3], '\n', "Tournament size: ", best_result[0][4], '\n', "Selected individuum: ", best_result[1], '\n', "Fitness for train data: ", best_result[2], '\n', "Average fitness with these parameters: ", best_result[3], '\n',  "Standard deviation: ", best_result[4], '\n', "Fitness for test data: ", fitness(best_classifier, test_data))
