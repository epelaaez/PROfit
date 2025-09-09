#ifndef PRO_TO_CALL_H
#define PRO_TO_CALL_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <stdexcept>

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"

// Root includes
namespace PROfit{


       /* Function: given a value for true variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the true value is out of range
     */
    int FindLocalVariableBin(const PROconfig &inconfig, float other_value, int channel_index, int other_index);

    /* Function: given a value for true or reconstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the true value is out of range, then return value of -1
     */
    int FindGlobalVariableBin(const PROconfig &inconfig, float other_value, int subchannel_index, int other_index);
    int FindGlobalVariableBin(const PROconfig &inconfig, float other_value, const std::string& subchannel_fullname, int other_index);


    /* Function: given a global bin index in the full vector, return the index of the subchannle this bin belongs to
     * Parameter:
     * 	 	inconfig:     a reference to PROconfig object, needed for calculating index 
     * 	 	global_bin:   global bin index. It can be a global true bin index, or global reco bin index
     * 	 		      Default to true.
     */
    int FindSubchannelIndexFromVariableGlobalBin(const PROconfig &inconfig, int global_bin, int var_index=0);

    /* Function: given a full matrix, collapse the matrix */
    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix);

    /* Function: given a full vector (that contains reco), collapse the vector */
    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector);

    /* Function: given a full matrix, collapse the matrix */
    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix, int other_index);

    /* Function: given a full vector (that contains reco), collapse the vector */
    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector, int other_index);

    std::string getIcon();
    
    /* Mostly a debug function to print all information from PROconfig on variables and the binning above. */
    void PrintVariableInfo(const PROconfig &inconfig);


    template <typename T>
    std::string to_string_prec(const T a_value, const int n = 6)    {
      std::ostringstream out;
      out <<std::fixed<< std::setprecision(n) << a_value;
      return out.str();
    }

    Eigen::MatrixXf ComputeSquareRootCovariance(const Eigen::MatrixXf& covariance, bool use_ldlt = true, float svd_tol = -1.0);

};

#endif
