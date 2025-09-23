#ifndef PROCONFIG_H_
#define PROCONFIG_H_

// STANDARD
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

// TINYXML2
#include "tinyxml2.h"

// EIGEN
#include <Eigen/Eigen>

//PROfit
#include "PROlog.h"
#include "MurmurHash3.h"

//ROOT
#include "TTreeFormula.h"
#include "TColor.h"


/*eweight_type here to switch between uboone style "float" and SBNcode style "float"  */
//#define TYPE_FLOAT
#ifdef TYPE_FLOAT  
typedef float eweight_type;
#else
typedef double eweight_type;
#endif

namespace PROfit{

    /*typedef the base "eventweight_class" branch that reweightable systematics are stored in in uboonestyle systematics 
    */
    typedef std::map<std::string, std::vector<eweight_type>> eweight_map;


    /* Struct: Branch variable is a SBNfit era class to load using TTReeFormula a givem variable (or function of variables) 
     * Note: was originally split between float/int, but moved to TTreeFormula
     */

    struct BranchVariable{
      std::string associated_hist;
      std::string associated_systematic;
      bool central_value;

      // Return type from a BranchVariable. Can be a single number or a list
      struct Value {
          std::vector<float> v;

          Value() {}
          Value(const std::vector<float> &v): v(v) {}
          Value(float v): v({v}) {}

          // Coerce the Value object into a single number
          float first() {return v[0];}
          size_t size() {return v.size();}
      };

      // Interface for loading data from an input. 
      // Define this as abstract so we can load data from different inputs 
      // in the future
      class Formula {
        public:
          virtual Value EvalInstance() = 0;
          virtual void LoadEvent(unsigned evtno) = 0;
          virtual std::string FormulaName() const = 0;
          virtual ~Formula() {}
      };

      std::shared_ptr<Formula> branch_monte_carlo_weight_formula = nullptr;
      std::vector<std::shared_ptr<Formula>> branch_variable_formulas;

      int model_rule;
      int include_systematics;

      std::vector<std::string> variable_names;

      bool hist_reweight;

      //constructor
      BranchVariable(std::string a) : associated_hist(a), central_value(false), model_rule(-9), include_systematics(1), hist_reweight(false){}
      BranchVariable(std::string a_hist, std::string a_syst, bool cv) :  associated_hist(a_hist), associated_systematic(a_syst), central_value(cv), model_rule(-9), include_systematics(1), hist_reweight(false){}
 

      void SetModelRule(const std::string & model_rule_def){model_rule = std::stoi(model_rule_def); return;}
      void SetIncludeSystematics(int insyst){include_systematics = insyst; return;} 

      //main loader for all variables
      void AddVariable(const std::string& var_in){ variable_names.push_back(var_in); return;}
     
      void SetReweight(bool inbool){ hist_reweight = inbool; return;}
      bool GetReweight() const {return hist_reweight;}

       int GetModelRule() const{
	return model_rule;
      };

      int GetIncludeSystematics() const{
	return include_systematics;
      };


      // Function: evaluate additional weight setup in the branch and return in floating precision 
      // Note: if no additional weight is set, value of 1.0 will be returned.
      inline
      float GetMonteCarloWeight() const{
	if(branch_monte_carlo_weight_formula){ 

	  return branch_monte_carlo_weight_formula->EvalInstance().first();
	}
	return 1.0;
      }

      std::vector<BranchVariable::Value> GetVariables() const {
          std::vector<BranchVariable::Value> ret;
          for(const auto &formula: branch_variable_formulas) {
              ret.push_back(formula->EvalInstance());
          }
          return ret;
      }


    };


    // Implementation of Formula interface for ROOT TTree, TTreeFormula based input
    class ROOTFormula: public BranchVariable::Formula {
      public:
        ROOTFormula(const std::string &name, const std::string &formula, TTree *t);
        BranchVariable::Value EvalInstance() override;
        void LoadEvent(unsigned eventno) override;
        std::string FormulaName() const override;
        virtual ~ROOTFormula() {}

      private:
        std::vector<std::unique_ptr<TTreeFormula>> fs;
        int treeNumber;
    };
 

    /* 
     * Class: Primary booking class for loading in info from XML and saving all information on mode_detector_channel_subchannel information.
     * Note:
     *  PROconfig to be created once and only once per PROfit executable and then passed by reference to various more complex functions, and ignored when not needed. 
     *  Contains information and collapsing matricies for collapsing subchannels->channels also. Filled once, on construction and then used by later classes when needed.
     */


    class PROconfig {
        private:

            //indicator of whether each channel/detector/subchannel is used
            std::vector<bool> m_mode_bool;
            std::vector<bool> m_detector_bool;
            std::vector<bool> m_channel_bool;
            std::vector<std::vector<bool>>  m_subchannel_bool;


            //map from subchannel name/index to global index and channel index
            std::unordered_map<std::string, size_t> m_map_fullname_subchannel_index;
            std::vector<size_t> m_vec_subchannel_index; //vector of global subchannel index, in increasing order
            std::vector<size_t> m_vec_channel_index;    //vector of corresponding channel index
            std::vector<std::vector<size_t>> m_vec_global_variable_index_start;  //vector of global true bin index, in increasing order

            //---- PRIVATE FUNCTION ------

            /* Function: construct a matrix T, which will be used to collapse matrix and vectors */
            void construct_variable_collapsing_matrices();

            /* Function: remove any mode/detector/channel/subchannels in the configuration xml that are not used from consideration
            */
            void remove_unused_channel();


            /* Function: ignore any file that is associated with unused channels 
            */
            void remove_unused_files();


            /* Function: fill in mapping between subchannel name/index to global indices */
            void generate_index_map();


            /* Function: given an input vector that's sorted in ascending order, and input val, return the index of elmeent which is equal to val
             * Note: it gives exception when input value is not present in the vector 
             */
            size_t find_equal_index(const std::vector<size_t>& input_vec, size_t val) const;

            /* Function: given an input vector that's sorted in ascending order, and input val, return the index of the closest element which is equal or smaller than val */
            size_t find_less_or_equal_index(const std::vector<size_t>& input_vec, size_t val) const;

            /* Function: given global bin index, return associated global subchannel index 
             * Note: not used anymore 
             */
            size_t find_global_subchannel_index_from_global_bin(size_t global_index, const std::vector<size_t>& num_subchannel_in_channel, const std::vector<size_t>& num_bins_in_channel, size_t num_channels, size_t num_bins_total) const;


        public:

            PROconfig() {}; //always have an empty constructor?

            /* Constructor Function: Need a string passed which is the filename (with path) of the configuration xml */
            PROconfig(const std::string &xmlname,bool rate_only=false);

            /*
             * Function: Use TinyXML2 to load XML */
            int LoadFromXML(const std::string & filename);
            uint32_t hash;

            static bool SameChannels(const PROconfig &one, const PROconfig &two);

            // Helper object. Represents a general binning in N dimensions
            struct Binning {
              std::vector<std::vector<float>> bin_edges;
              Binning() {}
              Binning(const std::vector<std::vector<float>> &b): bin_edges(b) {}
              Binning(const std::vector<float> &b): bin_edges({b}) {}

              // Helper functions
              size_t NDim() const {return bin_edges.size();}
              // Total number of bins
              size_t NBins() const;

              // Project an input index across the full binning to a 1D index across the input dimension
              size_t ProjectIndex(size_t ind, size_t dim) const;
              Eigen::VectorXf ProjectSpectra(const Eigen::VectorXf &in, size_t dim) const;
              Eigen::VectorXf ProjectSpectraErrors(const Eigen::VectorXf &in, size_t dim) const;

              // Bin an input value
              int Bin(const std::vector<float> &v) const;
              int Bin(float v) const {return Bin(std::vector<float>({v}));}
              int Bin(const BranchVariable::Value &v) const {return Bin(v.v);}

              // number of bins along a dimension
              size_t NBinsAlong(unsigned dim) const {return (bin_edges[dim].size() == 0) ? 0 : bin_edges[dim].size()-1;}
              // number of bin edges along a dimension
              size_t NBinEdgesAlong(unsigned dim) const {return bin_edges[dim].size();}
              // return edges along a dimension
              std::vector<float> Edges(unsigned dim = 0) const {return bin_edges[dim];}
              // return widths along a dimension
              std::vector<float> Widths(unsigned dim = 0) const;
            };

            std::string m_xmlname;	
            std::vector<double> m_det_pot;
            std::vector<std::string> m_fullnames;

            size_t m_num_detectors;
            size_t m_num_channels;
            size_t m_num_modes;
            size_t m_num_variables;

            std::vector<size_t> m_num_subchannels; 

            size_t i_prime;// The main fitting variable, will move to be xml config soon

            // New
            std::vector<bool> m_channel_variable_plot_bool;
            std::vector<std::vector<PROconfig::Binning>> m_channel_variable_bins;

            //the xml names are the way we track which channels and subchannels we want to use later
            std::vector<std::string> m_mode_names; 			
            std::vector<std::string> m_mode_plotnames; 			

            std::vector<std::string> m_detector_names; 		
            std::vector<std::string> m_detector_plotnames; 		

            std::vector<std::string> m_channel_names; 		
            std::vector<std::string> m_channel_plotnames; 		
            std::vector<std::string> m_channel_units; 		

            std::vector<std::vector<std::string>> m_channel_variable_units;

            std::vector<std::vector<std::string >> m_subchannel_names; 
            std::vector<std::vector<std::string >> m_subchannel_plotnames; 
            std::vector<std::vector<std::string >> m_subchannel_colors; 
            std::vector<std::vector<size_t >> m_subchannel_datas; 

            std::vector<size_t> m_num_variable_bins_detector_block;
            std::vector<size_t> m_num_variable_bins_mode_block;
            std::vector<size_t> m_num_variable_bins_total;
            std::vector<size_t> m_num_variable_bins_detector_block_collapsed; 
            std::vector<size_t> m_num_variable_bins_mode_block_collapsed; 
            std::vector<size_t> m_num_variable_bins_total_collapsed; 

            std::vector<Eigen::MatrixXf> variable_collapsing_matrices; 
            std::vector<Eigen::VectorXf> collapsed_bin_widths;

            //This section entirely for montecarlo generation of a covariance matrix or PROspec 
            bool m_write_out_variation;
            bool m_form_covariance;
            std::string m_write_out_tag;
            int m_num_variation_type_covariance = 0;
            int m_num_variation_type_spline = 0;
            int m_num_variation_type_flat = 0;
            int m_num_variation_type_norm = 0;

            int m_num_mcgen_files;
            std::vector<std::string> m_mcgen_tree_name;	
            std::vector<std::string> m_mcgen_file_name;	
            std::vector<long int> m_mcgen_maxevents;	
            std::vector<float> m_mcgen_pot;	
            std::vector<float> m_mcgen_scale;	
            std::vector<int> m_mcgen_numfriends;	
            std::vector<bool> m_mcgen_fake;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_map;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_treename_map;
            std::vector<std::vector<std::string>> m_mcgen_additional_weight_name;
            std::vector<std::vector<bool>> m_mcgen_additional_weight_bool;
            std::vector<std::vector<std::shared_ptr<BranchVariable>>> m_branch_variables;
            std::vector<std::vector<std::string>> m_mcgen_eventweight_branch_names;
            std::vector<std::vector<int>> m_mcgen_eventweight_branch_syst;

            //specific bits for covairiancegeneration
            bool m_use_mcstats = false;
            std::vector<std::string> m_mcgen_weightmaps_formulas;
            std::vector<bool> m_mcgen_weightmaps_uses;
            std::vector<std::string> m_mcgen_weightmaps_patterns;
            std::vector<std::string> m_mcgen_weightmaps_mode;
            std::vector<std::string> m_mcgen_variation_allowlist;
            std::vector<std::string> m_mcgen_variation_denylist;
            std::vector<std::string> m_mcgen_variation_type;
            std::map<std::string, std::string> m_mcgen_variation_type_map;
            std::map<std::string, std::string> m_mcgen_variation_plotname_map;
            std::map<std::string, int> m_mcgen_variation_binning_map;
            std::map<std::string, std::vector<double>> m_mcgen_variation_knobval_override;
            std::map<std::string, std::vector<std::string>> m_mcgen_variation_tags;
            std::map<std::string, std::vector<std::string>> m_mcgen_shapeonly_listmap; //a map of shape-only systematic and corresponding subchannels
            std::vector<std::tuple<std::string, std::string, float>> m_mcgen_correlations;
            std::map<std::string, float> m_mcgen_variation_prior;
            std::map<std::string, float> m_mcgen_variation_prior_centers;
      
            //FIX skepic
            std::vector<std::string> systematic_name;

            //Some model infomation
            std::string m_model_tag;
            std::vector<int> m_model_rule_index;
            std::vector<std::string> m_model_rule_names;
            std::vector<int> m_model_parameter_index;
            std::vector<std::string> m_model_parameter_names;
            std::map<std::string,int> m_model_parameter_map;

            bool m_bool_rate_only;
            //----- PUBLIC FUNCTIONS ------
            //


            /* Function: return matrix T, of size (m_num_bins_total, m_num_bins_total_collapsed), which will be used to collapse matrix and vectors 
             * Note: To collapse a full matrix M, please do T.transpose() * M * T
             * 	     To collapse a full vector V, please do T.transpose() * V
             */
            inline 
                Eigen::MatrixXf GetCollapsingMatrix() const {return variable_collapsing_matrices[i_prime]; }
            inline
                Eigen::MatrixXf GetCollapsingMatrix(int other_index) const {return variable_collapsing_matrices[other_index]; }

            /* Function: Calculate how big each mode block and decector block are, for any given number of channels/subchannels, before and after the collapse
             * Note: only consider mode/detector/channel/subchannels that are actually used 
             */
            void CalcTotalBins();


            /* Function: given subchannel full name, return global subchannel index 
             * Note: index start from 0, not 1
             */
            size_t GetSubchannelIndex(const std::string& fullname) const;

            /* Function: given global index (in the full vector), return global subchannel index of associated subchannel
             * Note: returns a 0-based index 
             */
            size_t GetSubchannelIndexFromVariableGlobalBin(size_t global_index, size_t var_index) const;

            /* Function: given subchannel global index, return corresponding channel index 
             * Note: index start from 0, not 1
             */
            size_t GetLocalChannelIndexFromGlobalSubchannelIndex(size_t global_subchannel_index) const;

            /* Function: Given a global channel index return the local channel index */
            size_t GetLocalChannelIndexFromGlobalChannelIndex(size_t global_channel_index) const; 


            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            size_t GetGlobalVariableBinStart(size_t subchannel_index, size_t other_index) const;

            size_t GetCollapsedGlobalVariableBinStart(size_t channel_index, size_t other_index) const;

            /* Function: given channel index, return list of bin edges for this channel */
            const Binning& GetChannelVariableBins(size_t channel_index, size_t other_index) const;

            /* Function: Hex to int*/
            int HexToROOTColor(const std::string& hexColor) const;

            /* Calculate hash of unique properties of XML config for PROpeller */
            uint32_t CalcHash() const;
                    

    };
}
#endif
