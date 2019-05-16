/*
PROJECT DESCRIPTION:
Taint analysis tracks information flows from an object x(source) to another object y(sink), 
whenever information stored in x is transferred to object y. In this task, you will write an 
LLVM  pass to perform the taint analysis. In order to handle loops, the designed analysis 
 should be continued until a fixpoint is reached. 
*/

#include <cstdio>
#include <iostream>
#include <set>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/CFG.h"

using namespace llvm;

void printAnalysisMap(std::map<std::string,std::set<Instruction*>> analysisMap);
std::string getSimpleNodeLabel(const BasicBlock *Node);

int main(int argc, char **argv)
{
    // Read the IR file.
  static LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(argv[1], Err, Context);
  if (M == nullptr)
  {
    fprintf(stderr, "error: failed to load LLVM IR file \"%s\"", argv[1]);
    return EXIT_FAILURE;
  }

  // map the label name of each Basic Block to a set of tainted variables
  for (auto &F: *M) {
      if (strncmp(F.getName().str().c_str(),"main",4) == 0){
        std::map<std::string,std::set<Instruction*>> analysisMap;
        // continue analysis until a fixed point is reached to handle loops
        bool hasChange = true;
        while (hasChange) {
          hasChange = false;
          // iterate through each basic block
          for (auto &BB: F) {
            std::set<Instruction*> bSet = analysisMap[getSimpleNodeLabel(&BB)];
            auto initialSize = bSet.size();
            // retrieve set of tainted variables of its predecessor blocks and 
            // insert them into current basic block set
            for (BasicBlock *PB : predecessors(&BB)) {
              std::set<Instruction*> pSet = analysisMap[getSimpleNodeLabel(PB)];
              bSet.insert(pSet.begin(), pSet.end());
            }
            // copy current basic block set into temp set
            std::set<Instruction*> tempSet(bSet);
            // iterate through instructions of the basic block
            for (auto &I: BB) {
              if (isa<AllocaInst>(I)) {
                // insert source variable into temp and basic block set
                if (strncmp(I.getName().str().c_str(),"source",6) == 0) {
                  tempSet.insert(&I);
                  bSet.insert(&I);
                }
              } else if (isa<LoadInst>(I)) {
                // if loading a tainted value into a register, insert the register 
                // into temp set
                Value* v = I.getOperand(0);
                Instruction* var = dyn_cast<Instruction>(v);
                if (tempSet.count(var) == 1) {
                  tempSet.insert(&I);
                }
              } else if (I.isBinaryOp()) {
                // if either of the operands in a binary operation is tainted, 
                // insert the register into temp set
                for (int a = 0; a < 2; a++) {
                  Value* v = I.getOperand(a);
                  Instruction* var = dyn_cast<Instruction>(v);
                  if (tempSet.count(var) == 1) {
                    tempSet.insert(&I);
                  }
                }
              } else if (isa<StoreInst>(I)) {
                // if storing into a variable, erase the variable from temp and 
                // basic block set if it was previously tainted
                Value* v = I.getOperand(0);
                Instruction* var = dyn_cast<Instruction>(v);
                Value* v1 = I.getOperand(1);
                Instruction* var1 = dyn_cast<Instruction>(v1);
                tempSet.erase(var1);
                bSet.erase(var1);
                // if storing a tainted register into a variable, insert the 
                // variable into temp and basic block set
                if (tempSet.count(var) == 1) {
                  tempSet.insert(var1);
                  bSet.insert(var1);
                }
              }
            }
            // check if basic block's instruction set has changed
            if (bSet.size() != initialSize) 
                hasChange = true;
            analysisMap[getSimpleNodeLabel(&BB)] = bSet;
          }
        }
        printAnalysisMap(analysisMap);
      }
    }
    return 0;
  }

// Printing the results of the analysis map
void printAnalysisMap(std::map<std::string,std::set<Instruction*>> analysisMap) {
  for (auto& row : analysisMap){
    std::set<Instruction*> initializedVars = row.second;
    std::string BBLabel = row.first;
    errs() << BBLabel << ":\n";
    for (Instruction* var : initializedVars){
      errs() << "\t";
      var->dump();
    }
    errs() << "\n";
    }
}

// Printing Basic Block Label 
std::string getSimpleNodeLabel(const BasicBlock *Node) {
  if (!Node->getName().empty())
      return Node->getName().str();
  std::string Str;
  raw_string_ostream OS(Str);
  Node->printAsOperand(OS, false);
  return OS.str();
}
