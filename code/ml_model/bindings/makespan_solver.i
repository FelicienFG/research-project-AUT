%module makespan_solver

%{
#define SWIG_FILE_WITH_INIT
#include "../cpp/makespan_solver.h"
%}

%include "std_vector.i"
%include "std_list.i"
// Instantiate templates used
namespace std {
   %template(IntVector) vector<int>;
   %template(DagSubtaskVector) vector<DagSubtask>;
   %template(IntList) list<int>;
}

%include "../cpp/makespan_solver.h"