%module makespan_solver

%{
#define SWIG_FILE_WITH_INIT
#include "../cpp/makespan_solver.h"
%}

%include "std_vector.i"
// Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
}

%include "../cpp/makespan_solver.h"