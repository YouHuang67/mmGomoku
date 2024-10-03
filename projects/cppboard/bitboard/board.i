%module board
%{
#include "board_wrapper.h"
%}

%include "std_vector.i"
namespace std
{
	%template(IntVector) vector<int>;
}

%include "board_wrapper.h"