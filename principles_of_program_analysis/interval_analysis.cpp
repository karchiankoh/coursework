/*
PROJECT DESCRIPTION:
In this approach, an interval for a variable x is represented by x : [a;b] where a is a 
lower bound and a b is an upper bound on the possible values that x can contain. In this 
part, you will be implementing an Interval analysis with some path sensitivity for all the 
variables in the input program. In order to guarantee termination, the analysis may now not 
be able to produce a finite bound at all times.
*/

#include <cstdio>
#include <iostream>
#include <set>
#include <map>
#include <stack>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"

#include <limits>
#include <cstdlib>

using namespace llvm;
#define POS_INF std::numeric_limits<int>::max()
#define NEG_INF std::numeric_limits<int>::min()
typedef std::map<Value*,std::pair<int, int> > BBANALYSIS;

std::map<std::string,BBANALYSIS > analysisMap;
std::set<std::string> path;
std::map<std::string,BBANALYSIS > abstractAnalysisMap;
std::set<int> abstractDomain = { NEG_INF, -20, -10, -2, -1, 0, 1, 5, 6, 10, 20, POS_INF};
std::map<std::string,BBANALYSIS > copiedAnalysisMap;
std::string whileEndLabel = "%14";

// Printing Basic Block Label 
std::string getSimpleNodeLabel(const BasicBlock *Node) {
  if (!Node->getName().empty())
      return Node->getName().str();
  std::string Str;
  raw_string_ostream OS(Str);
  Node->printAsOperand(OS, false);
  return OS.str();
}

std::pair<int, int> getAbstractInterval(std::pair<int, int> oldInterval) {
	std::pair<int, int> newInterval;
	for ( auto it = abstractDomain.begin();it != abstractDomain.end(); ++it) {
		if (oldInterval.first >= *it)
			newInterval.first = *it;
		if (oldInterval.second <= *it) {
			newInterval.second = *it;
			break;
		}
	}
	return newInterval;
}

void makeAbstract() {
	abstractAnalysisMap.clear();
	abstractAnalysisMap.insert(analysisMap.begin(), analysisMap.end());
	for ( auto it = abstractAnalysisMap.begin();it != abstractAnalysisMap.end(); ++it)
	{
		BBANALYSIS analysis = it->second;
		for ( auto it1 = analysis.begin();it1 != analysis.end(); ++it1) {
			analysis[it1->first] = getAbstractInterval(it1->second);
		}
		abstractAnalysisMap[it->first] = analysis;
	}
}

//======================================================================
// Check fixpoint reached
//======================================================================
bool fixPointReached(std::map<std::string,BBANALYSIS> oldAnalysisMap) {
	if (oldAnalysisMap.empty())
		return false;
	for ( auto it = abstractAnalysisMap.begin();it != abstractAnalysisMap.end(); ++it)
	{
		if(oldAnalysisMap[it->first] != it->second)
			return false;
	}
	return true;
}

// Performs analysis union
BBANALYSIS union_analysis(BBANALYSIS A, BBANALYSIS B)
{
	for ( auto it = A.begin();it != A.end(); ++it)
	{
		if (B.count(it->first) > 0) {
			std::pair<int, int> aSet = A[it->first];
			std::pair<int, int> bSet = B[it->first];
			std::pair<int, int> set;
			if (bSet.first < aSet.first) {
				set.first = bSet.first;
			} else {
				set.first = aSet.first;
			}
			if (bSet.second > aSet.second) {
				set.second = bSet.second;
			} else {
				set.second = aSet.second;
			} 
			A[it->first] = set;
		}
	}

	for ( auto it = B.begin(); it != B.end(); ++it)
	{
		if (A.count(it->first) == 0) 
			A[it->first] = B[it->first]; 
	}

    return A;
}

//======================================================================
// update Basic Block Analysis
//======================================================================

// Processing Alloca Instruction
std::pair<int, int> processAlloca()
{
	std::pair<int, int> set;
	set.first = NEG_INF;
	set.second = POS_INF;
	return set;
}

std::pair<int, int> retrieveSet(Value* op1, BBANALYSIS analysis)
{
	if(isa<ConstantInt>(op1)){
		llvm::ConstantInt *CI = dyn_cast<ConstantInt>(op1);
		int64_t op1Int = CI->getSExtValue();
		std::pair<int, int> set;
		set.first = op1Int;
		set.second = op1Int;
		return set;
	}else if (analysis.find(op1) != analysis.end() ){
		return analysis[op1];
	} else {
		std::pair<int, int> set;
		set.first = NEG_INF;
		set.second = POS_INF;
		return set;
	}
}

// Processing Store Instruction
std::pair<int, int> processStore(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	return retrieveSet(op1, analysis);
}

// // Processing Load Instruction
std::pair<int, int> processLoad(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	return retrieveSet(op1, analysis);
}

int multiply(int op1, int op2) {
	if (op1 == 0 || op2 == 0)
		return 0;
	if (op1 == NEG_INF) {
		if (op2 < 0)
			return POS_INF;
		else
			return NEG_INF;
	}
	if (op2 == NEG_INF) {
		if (op1 < 0)
			return POS_INF;
		else
			return NEG_INF;
	}
	if (op1 == POS_INF) {
		if (op2 < 0)
			return NEG_INF;
		else
			return POS_INF;
	}
	if (op2 == POS_INF) {
		if (op1 < 0)
			return NEG_INF;
		else
			return POS_INF;
	}
	return op1 * op2;
}

// Processing Mul Instructions
std::pair<int, int> processMul(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	Value* op2 = I->getOperand(1);
	std::pair<int, int> set1, set2;
	set1 = retrieveSet(op1, analysis);
	set2 = retrieveSet(op2, analysis);

	int lower = multiply(set1.first, set2.first);
	int upper = multiply(set1.second, set2.second);

	std::pair<int, int> set;
	set.first = lower;
	set.second = upper;
	return set;
}

int divide(int op1, int op2) {
	if (op2 == POS_INF || op2 == NEG_INF)
		return 0;
	if (op1 == POS_INF)
		return POS_INF;
	if (op1 == NEG_INF)
		return NEG_INF;
	if (op2 == 0) {
		if (op1 < 0)
			return NEG_INF;
		else 
			return POS_INF;
	}
	return op1 / op2;
}

// Processing Div Instructions
std::pair<int, int> processDiv(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	Value* op2 = I->getOperand(1);
	std::pair<int, int> set1, set2;
	set1 = retrieveSet(op1, analysis);
	set2 = retrieveSet(op2, analysis);

	int lower = divide(set1.first, set2.second);
	int upper = lower;
	int temp = divide(set1.second, set2.first);
	if (temp < lower)
		lower = temp;
	if (temp > upper)
		upper = temp;

	std::pair<int, int> set;
	set.first = lower;
	set.second = upper;
	return set;
}

int add(int op1, int op2) {
	if (op1 == NEG_INF || op2 == NEG_INF)
		return NEG_INF;
	if (op1 == POS_INF || op2 == POS_INF)
		return POS_INF;
	return op1 + op2;
}

// Processing Add Instructions
std::pair<int, int> processAdd(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	Value* op2 = I->getOperand(1);
	std::pair<int, int> set1, set2;
	set1 = retrieveSet(op1, analysis);
	set2 = retrieveSet(op2, analysis);

	int lower = add(set1.first, set2.first);
	int upper = add(set1.second, set2.second);

	std::pair<int, int> set;
	set.first = lower;
	set.second = upper;
	return set;
}

int subtract(int op1, int op2) {
	if (op1 == NEG_INF)
		return NEG_INF;
	if (op1 == POS_INF)
		return POS_INF;
	if (op2 == NEG_INF)
		return POS_INF;
	if (op2 == POS_INF)
		return NEG_INF;
	return op1 - op2;
}

// Processing Sub Instructions
std::pair<int, int> processSub(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	Value* op2 = I->getOperand(1);
	std::pair<int, int> set1, set2;
	set1 = retrieveSet(op1, analysis);
	set2 = retrieveSet(op2, analysis);

	int lower = subtract(set1.first, set2.second);
	int upper = subtract(set1.second, set2.first);

	std::pair<int, int> set;
	set.first = lower;
	set.second = upper;
	return set;
}

// Processing Rem Instructions
std::pair<int, int> processRem(llvm::Instruction* I, BBANALYSIS analysis)
{
	Value* op1 = I->getOperand(0);
	Value* op2 = I->getOperand(1);
	std::pair<int, int> set1, set2;
	set1 = retrieveSet(op1, analysis);
	set2 = retrieveSet(op2, analysis);

	int lower, upper = 0;
	if (set1.second >= set2.second) {
		upper = set2.second - 1;
	} else if (set1.second >= set2.first) {
		if (set2.first - 1 > set1.second)
			upper = set2.first - 1;
		else
			upper = set1.second;
	} else {
		upper = set1.second;
	}
	if (set1.second >= set2.first || set1.first <= 0) 
		lower = 0;
	else 
		lower = set1.first;

	std::pair<int, int> set;
	set.first = lower;
	set.second = upper;
	return set;
}

// update Basic Block Analysis
BBANALYSIS updateBBAnalysis(BasicBlock* BB,BBANALYSIS analysis)
{
	BBANALYSIS temp;
	temp.insert(analysis.begin(), analysis.end());

	// Loop through instructions in BB
	for (auto &I: *BB)
	{
		if (isa<AllocaInst>(I)){
			if (I.getName().str() != "retval") {
				analysis[&I] = processAlloca();
				temp[&I] = analysis[&I];
			}
		}else if (isa<StoreInst>(I)){
			Value* op2 = I.getOperand(1);
			if (op2->getName().str() != "retval") {
				analysis[op2] = processStore(&I, temp);
				temp[op2] = analysis[op2];
			}
	    }else if(isa<LoadInst>(I)){
	    	Value* op1 = I.getOperand(0);
			if (op1->getName().str() != "retval") {
				temp[&I] = processLoad(&I, temp);
			}
	    }else if(I.getOpcode() == BinaryOperator::SDiv){
			temp[&I] = processDiv(&I, temp);
	    }else if(I.getOpcode() == BinaryOperator::Mul){
			temp[&I] = processMul(&I, temp);
	    }else if(I.getOpcode() == BinaryOperator::Add){
	    	temp[&I] = processAdd(&I, temp);
	    }else if(I.getOpcode() == BinaryOperator::Sub){
			temp[&I] = processSub(&I, temp);
	    }else if(I.getOpcode() == BinaryOperator::SRem){
	    	temp[&I] = processRem(&I, temp);
	    }
    }
	return analysis;
}

//======================================================================
// update Graph Analysis
//======================================================================
BBANALYSIS applyCond_aux(BBANALYSIS predSet, Instruction* I, std::pair<int,int> set){
	if (isa<AllocaInst>(I)){
    	predSet[I] = set;
    }else if(isa<LoadInst>(I)){
    	predSet[I] = set;
    	predSet = applyCond_aux(predSet,dyn_cast<Instruction>(I->getOperand(0)),set);
    }
	return predSet;
}

// Apply condition to the predecessor set
BBANALYSIS applyCond(BBANALYSIS predSet, BasicBlock* predecessor, BasicBlock* BB)
{
	for (auto &I: *predecessor)
	{
		if (isa<BranchInst>(I)){
			BranchInst* br = dyn_cast<BranchInst>(&I);
			if(!br->isConditional()) {
				return predSet;
			}
			llvm::CmpInst *cmp = dyn_cast<llvm::CmpInst>(br->getCondition());
			Value* op1 = cmp->getOperand(0);
			Value* op2 = cmp->getOperand(1);
			std::pair<int,int> set1,set2;

			if(isa<ConstantInt>(op1)){
				llvm::ConstantInt *CI = dyn_cast<ConstantInt>(op1);
				int64_t op1Int = CI->getSExtValue();
				set1.first = op1Int;
				set1.second = op1Int;
			}else if(isa<LoadInst>(dyn_cast<Instruction>(op1))) {
				Value* allocaOp1 = dyn_cast<Instruction>(op1)->getOperand(0);
				if (predSet.find(allocaOp1) != predSet.end() )
					set1 = predSet[allocaOp1];
			} else {
				set1.first = NEG_INF;
				set1.second = POS_INF;
			}

			if(isa<ConstantInt>(op2)){
				llvm::ConstantInt *CI = dyn_cast<ConstantInt>(op2);
				int64_t op2Int = CI->getSExtValue();
				set2.first = op2Int;
				set2.second = op2Int;
			}else if(isa<LoadInst>(dyn_cast<Instruction>(op2))) {
				Value* allocaOp2 = dyn_cast<Instruction>(op2)->getOperand(0);
				if (predSet.find(allocaOp2) != predSet.end() )
					set2 = predSet[allocaOp2];
			} else {
				set2.first = NEG_INF;
				set2.second = POS_INF;
			}

			bool flag;
			if(BB == br->getOperand(2)) flag = true;
			if(BB == br->getOperand(1)) flag = false;

			int cmpValue;
			switch (cmp->getPredicate()) {
			  case llvm::CmpInst::ICMP_SGT:{
				  if(flag == true) cmpValue = 1;
				  else { cmpValue = 0;
				  }
				  break;
			  }
			  case llvm::CmpInst::ICMP_SLE:{
				  if(flag == false) cmpValue = 1;
				  else { cmpValue = 0;
				  }
				  break;
			  }
			  case llvm::CmpInst::ICMP_SGE:{
				  if(flag == true) cmpValue = 3;
				  else { cmpValue = 2;
				  }
				  break;
			  }
			  case llvm::CmpInst::ICMP_SLT:{
				  if(flag == false) cmpValue = 3;
				  else { cmpValue = 2;
				  }
				  break;
			  }
			  case llvm::CmpInst::ICMP_EQ:{
				  if(flag == true) cmpValue = 5;
				  else { cmpValue = 4;
				  }
				  break;
			  }
			  case llvm::CmpInst::ICMP_NE:{
				  if(flag == false) cmpValue = 4;
				  else { cmpValue = 5;
				  }
				  break;
			  }
			}

			if(isa<LoadInst>(dyn_cast<Instruction>(cmp->getOperand(0)))){
				switch(cmpValue) {
					case 0: { // less than equals
						if (set1.first > set2.second) {
							path.erase(getSimpleNodeLabel(BB));
						}
						break;
					}
					case 1: { // greater than
						if (set1.second <= set2.first) {
							path.erase(getSimpleNodeLabel(BB));
						}
						break;
					}
					case 2: { // less than
						if (set1.first >= set2.second)
							path.erase(getSimpleNodeLabel(BB));
						break;
					}
					case 3: { // greater than equals
						if (set1.second < set2.first)
							path.erase(getSimpleNodeLabel(BB));
						break;
					}
					case 4: { // not equals
						if (set1.second == set2.first)
							path.erase(getSimpleNodeLabel(BB));
						break;
					}
					case 5: { // equals
						if (set1.second != set2.first)
							path.erase(getSimpleNodeLabel(BB));
						break;
					}
				}
			}
		}
	}

	return predSet;
}

// update Graph Analysis
void updateGraphAnalysis(Function *F, int i) {
	bool isInitBlock = true;
    for (auto &BB: *F){
    	if (i == 0 || !isInitBlock)
    		path.insert(getSimpleNodeLabel(&BB));

    	BBANALYSIS predUnion;
        // Load the current stored analysis for all predecessor nodes
    	for (auto it = pred_begin(&BB), et = pred_end(&BB); it != et; ++it)
    	{
    		BasicBlock* predecessor = *it;
	    	BBANALYSIS predSet = applyCond(analysisMap[getSimpleNodeLabel(predecessor)],predecessor, &BB);
	    	if (path.find(getSimpleNodeLabel(predecessor)) != path.end() ) {
	    		predUnion = union_analysis(predUnion,predSet);
    		}
    	}

    	BBANALYSIS BBAnalysis = updateBBAnalysis(&BB,predUnion);
    	analysisMap[getSimpleNodeLabel(&BB)] = BBAnalysis;
    	
    	isInitBlock = false;
    }

    for (auto &BB: *F){
    	BBANALYSIS BBAnalysis = analysisMap[getSimpleNodeLabel(&BB)];
    	BBANALYSIS OldBBAnalysis = copiedAnalysisMap[getSimpleNodeLabel(&BB)];
		analysisMap[getSimpleNodeLabel(&BB)] = union_analysis(BBAnalysis,OldBBAnalysis);
	}
}

//======================================================================
// main function
//======================================================================

int main(int argc, char **argv)
{
    // Read the IR file.
    static LLVMContext Context;
    SMDiagnostic Err;

    // Extract Module M from IR (assuming only one Module exists)
    std::unique_ptr<Module> M = parseIRFile(argv[1], Err, Context);
    if (M == nullptr)
    {
      fprintf(stderr, "error: failed to load LLVM IR file \"%s\"", argv[1]);
      return EXIT_FAILURE;
    }

    // 1.Extract Function main from Module M
    Function *F = M->getFunction("main");

    // 2.Define analysisMap as a mapping of basic block labels to empty set (of instructions):
    // For example: Assume the input LLVM IR has 4 basic blocks, the map
    // would look like the following:
    // entry -> {}
    // if.then -> {}
    // if.else -> {}
    // if.end -> {}
    for (auto &BB: *F){
    	BBANALYSIS emptySet;
    	analysisMap[getSimpleNodeLabel(&BB)] = emptySet;
    }
    // Note: All variables are of type "alloca" instructions. Ex.
    // Variable a: %a = alloca i32, align 4

    // Keeping a snapshot of the previous ananlysis
    std::map<std::string,BBANALYSIS> oldAnalysisMap;

    // Fixpoint Loop
    int i = 0;
    while(!fixPointReached(oldAnalysisMap)){
        oldAnalysisMap.clear();
        oldAnalysisMap.insert(abstractAnalysisMap.begin(), abstractAnalysisMap.end());
	    updateGraphAnalysis(F, i);
	    makeAbstract();
		llvm::errs() << "Round:" << i++ <<  "\n";
		for ( auto it = abstractAnalysisMap.begin();it != abstractAnalysisMap.end(); ++it)
		{
			llvm::errs() << it->first << ":\n";
			BBANALYSIS analysis = it->second;
			for ( auto it1 = analysis.begin();it1 != analysis.end(); ++it1){
				std::pair<int, int> set1 = it1->second;
				llvm::errs() << "\t";
		    	it1->first->dump();
		    	llvm::errs() << "\t\t\t\t\t";
		    	llvm::errs() << "[" << set1.first << "," << set1.second << "]";
		    	llvm::errs() << "\n";
			}
		}
		llvm::errs() << "\n";

		copiedAnalysisMap.clear();
		copiedAnalysisMap.insert(analysisMap.begin(), analysisMap.end());
		analysisMap.clear();
		analysisMap[whileEndLabel] = copiedAnalysisMap[whileEndLabel];
		path.clear();
		path.insert(whileEndLabel);
	}

    return 0;
}
