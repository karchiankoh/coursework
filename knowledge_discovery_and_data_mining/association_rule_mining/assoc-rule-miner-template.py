'''
ASSIGNMENT DESCRIPTION: 
In this part, you are going to implement the association rule mining algorithm,
we have provided the data, the test cases, the template output and template
code to you. What you need to do in this part is to implement the gener-
ate frequent itemset(...) function and generate association rules(...) function
in the template file.
'''

import sys
import os
import itertools

masterSupportDict = {}

def cmp(a, b):
    return (a > b) - (a < b)

def read_csv(filepath):
	'''Read transactions from csv_file specified by filepath
	Args:
		filepath (str): the path to the file to be read

	Returns:
		list: a list of lists, where each component list is a list of string representing a transaction

	'''

	transactions = []
	with open(filepath, 'r') as f:
		lines = f.readlines()
		for line in lines:
			transactions.append(line.strip().split(',')[:-1])
	return transactions

def generate_frequent_1_itemsets(transactions, minsup):
	# Generate {frequent 1-itemsets}
	newList = []
	supportDict = {}
	maxTransactionLen = 0

	for transaction in transactions:
		if len(transaction) > maxTransactionLen:
			maxTransactionLen = len(transaction)
		for item in transaction:
			itemTuple = (item, )
			if supportDict.get(itemTuple) == None:
				supportDict[itemTuple] = 1
			else:
				supportDict[itemTuple] += 1

	numTransactions = len(transactions)
	tempList = []
	for itemTuple, count in supportDict.items():
		if count / numTransactions >= minsup:
			newList.append(list(itemTuple))

	newList.sort()
	masterSupportDict.update(supportDict)

	return newList, maxTransactionLen

def candidate_generation(k, oldList):
	# Generate L(k+1) from F(k)
	candidateList = []
	tempCGList = []

	if k==1:
		for x in range(0, len(oldList) ):
			for y in range(x+1, len(oldList) ):
				tempCGList = oldList[x][:] + oldList[y][:]
				candidateList.append(tempCGList)
	else:
		for x in range(0, len(oldList) ):
			for y in range(x+1, len(oldList) ):
				if oldList[x][:k-1] == oldList[y][:k-1] :
					tempCGList = oldList[x][:]
					tempCGList.append(oldList[y][k-1])
					candidateList.append(tempCGList)
				else:
					break

	return candidateList

def candidate_pruning(k, candidateList, oldList):
	# Prune candidate itemsets in L(k+1) containing subsets of length k that are infrequent		
	if k>1:
		candidateListCopy = candidateList[:]
		for candidate in candidateListCopy:
			isFrequent = True
			for x in range(0, k-1) :
				candidateCopy = candidate[:]
				candidateCopy.pop(x)
				for itemset in oldList:
					compare = cmp(itemset, candidateCopy)
					if compare == 0:
						break
					elif compare > 0:
						isFrequent = False
						break
				if not isFrequent:
					candidateList.remove(candidate)
					break

	return candidateList

def support_counting(candidateList, transactions, k):
	# Count the support of each candidate in L(k+1) by scanning the DB
	candidateDict = {} 

	for candidate in candidateList:
		candidateDict[tuple(candidate)] = 0
	for transaction in transactions: 
		if len(transaction) > k:
			transaction.sort()
			subsetList = list(itertools.combinations(transaction, k+1)) # should have only generated ordered subsets to reduce computation...
			for subset in subsetList:
				if candidateDict.get(subset) != None:
					candidateDict[subset] += 1

	masterSupportDict.update(candidateDict)
	return candidateDict

def candidate_elimination(transactions, candidateDict, minsup):
	# Eliminate candidates in L(k+1) that are infrequent, leaving only those that are frequent => F(k+1)
	newList = []

	numTransactions = len(transactions)
	for itemset, supportCount in candidateDict.items():
		if supportCount / numTransactions >= minsup:
			newList.append(list(itemset))
	newList.sort()

	return newList

# To be implemented
def generate_frequent_itemset(transactions, minsup):
	'''Generate the frequent itemsets from transactions
	Args:
		transactions (list): a list of lists, where each component list is a list of string representing a transaction
		minsup (float): specifies the minsup for mining

	Returns:
		list: a list of frequent itemsets and each itemset is represented as a list string

	Example:
		Output: [['margarine'], ['ready soups'], ['citrus fruit','semi-finished bread'], ['tropical fruit','yogurt','coffee'], ['whole milk']]
		The meaning of the output is as follows: itemset {margarine}, {ready soups}, {citrus fruit, semi-finished bread}, {tropical fruit, yogurt, coffee}, {whole milk} are all frequent itemset

	'''
	# Let k=1
	k = 1;

	newList, maxTransactionLen = generate_frequent_1_itemsets(transactions, minsup) # Generate F(1)

	frequent_itemset = newList[:]
	oldList = newList[:]
	newList = []

	# Repeat until F(k) is empty
	while (len(oldList)>0 and k<maxTransactionLen) :

		candidateList = candidate_generation(k, oldList) # Candidate Generation
		candidateList = candidate_pruning(k, candidateList, oldList) # Candidate Pruning
		candidateDict = support_counting(candidateList, transactions, k) # Support Counting
		newList = candidate_elimination(transactions, candidateDict, minsup) # Candidate Elimination

		frequent_itemset += newList[:]
		oldList = newList[:]
		newList = []

		k+=1
		
	return frequent_itemset

def prune(itemset, hList, minconf):
	outputRules = []
	temp = []
	newHList = hList[:]

	# for each h in H(m)
	for h in hList:

		# conf = c( f(k) - h -> h )
		itemsetCopy = itemset[:]
		for item in h:
			itemsetCopy.remove(item)
		conf = masterSupportDict.get(tuple(itemset)) / masterSupportDict.get(tuple(itemsetCopy))
		
		# if conf > = minconf, add f(k) - h -> h into the output rules
		if conf >= minconf:
			tempOutputList = itemsetCopy[:]
			tempOutputList.append("=>")
			tempOutputList += h
			outputRules.append(tempOutputList)
			
		# else, remove h from newHList
		else:
			newHList.remove(h)

	return newHList, outputRules

def ap_genrule(itemset, hList, minconf):
	agOutputRules = []

	# while k > m + 1
	k = len(itemset)
	m = len(hList[0])
	while k > m+1:

		hList = candidate_generation(m, hList) # H(m+1) = apriori-gen( H(m) )
		hList, outputRules = prune(itemset, hList, minconf) # Prune ( f(k), H(m+1) )
		agOutputRules += outputRules
		m += 1 # m += 1

	return agOutputRules

# To be implemented
def generate_association_rules(transactions, minsup, minconf):
	'''Mine the association rules from transactions
	Args:
		transactions (list): a list of lists, where each component list is a list of string representing a transaction
		minsup (float): specifies the minsup for mining
		minconf (float): specifies the minconf for mining

	Returns:
		list: a list of association rule, each rule is represented as a list of string

	Example:
		Output: [['root vegetables', 'rolls/buns','=>', 'other vegetables'],['root vegetables', 'yogurt','=>','other vegetables']]
		The meaning of the output is as follows: {root vegetables, rolls/buns} => {other vegetables} and {root vegetables, yogurt} => {other vegetables} are the two associated rules found by the algorithm
	

	'''
	frequent_itemset = generate_frequent_itemset(transactions, minsup)
	masterOutputRules = []
	
	# for each frequent itemset f(k), k >= 2
	for itemset in frequent_itemset:
		if len(itemset) == 1:
			continue

		# H(1) = { {x} x in f(k) }
		hList = []
		for x in itemset:
			hList.append([x])

		# H(1) = Prune( f(k), H(1) )
		hList, outputRules = prune(itemset, hList, minconf)
		masterOutputRules += outputRules

		# Ap-genrules( f(k), H(1) )
		if len(hList) > 0:
			agOutputRules = ap_genrule(itemset, hList, minconf)
			masterOutputRules += agOutputRules
	
	return masterOutputRules


def main():

	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print("Wrong command format, please follwoing the command format below:")
		print("python assoc-rule-miner-template.py csv_filepath minsup")
		print("python assoc-rule-miner-template.py csv_filepath minsup minconf")
		exit(0)

	
	if len(sys.argv) == 3:
		transactions = read_csv(sys.argv[1])
		result = generate_frequent_itemset(transactions, float(sys.argv[2]))

		# store frequent itemsets found by your algorithm for automatic marking
		with open('.'+os.sep+'Output'+os.sep+'frequent_itemset_result.txt', 'w') as f:
			for items in result:
				output_str = '{'
				for e in items:
					output_str += e
					output_str += ','

				output_str = output_str[:-1]
				output_str += '}\n'
				f.write(output_str)

	elif len(sys.argv) == 4:
		transactions = read_csv(sys.argv[1])
		minsup = float(sys.argv[2])
		minconf = float(sys.argv[3])
		result = generate_association_rules(transactions, minsup, minconf)

		# store associative rule found by your algorithm for automatic marking
		with open('.'+os.sep+'Output'+os.sep+'assoc-rule-result.txt', 'w') as f:
			for items in result:
				output_str = '{'
				for e in items:
					if e == '=>':
						output_str = output_str[:-1]
						output_str += '} => {'
					else:
						output_str += e
						output_str += ','

				output_str = output_str[:-1]
				output_str += '}\n'
				f.write(output_str)


main()