/*
PROJECT DESCRIPTION:
We define a simple analysis, initialized variables analysis, which returns the set of all the
 variables that are initialized at any program point. In this task, you will write an LLVM 
 pass (similar to the LLVM pass in StackSet.cpp which was explained in the demo session) to 
 perform the initialized variables analysis. In order to handle loops, the designed analysis 
 should be continued until a fixpoint is reached. 
*/

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

  // map the label name of each Basic Block to a set of initialized variables
  for (auto &F: *M) {
    if (strncmp(F.getName().str().c_str(),"main",4) == 0){
      std::map<std::string,std::set<Instruction*>> analysisMap;
      // continue analysis until a fixed point is reached to handle loops
      bool hasChange = true;
      while (hasChange) {
        bool hasRetVal = false;
        hasChange = false;
        for (auto &BB: F){
          std::set<Instruction*> bSet = analysisMap[getSimpleNodeLabel(&BB)];
          auto initialSize = bSet.size();
          // retrieve set of initialized variables at predecessor block
          for (BasicBlock *PB : predecessors(&BB)) {
            std::set<Instruction*> pSet = analysisMap[getSimpleNodeLabel(PB)];
            bSet.insert(pSet.begin(), pSet.end());
          }
          // loop through instructions in a basic block and look for store 
          // instructions with alloca instruction as second argument
          for (auto &I: BB) {
            if (isa<StoreInst>(I)) {
              // load second argument's instruction and then cast it to a 
              // LLVM instruction
              Value* v = I.getOperand(1); // retrieving second argument
              Instruction* var = dyn_cast<Instruction>(v);
              if (isa<AllocaInst>(v)) {
                if (hasRetVal)
                  bSet.insert(var);
                else
                  hasRetVal = true; // do not insert retval variable into set
              }
            }
          }
          if (bSet.size() > initialSize)
            hasChange = true;
          analysisMap[getSimpleNodeLabel(&BB)] = bSet;
        }
      }
      printAnalysisMap(analysisMap); // print analysis map
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
