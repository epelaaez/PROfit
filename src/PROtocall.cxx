#include "PROtocall.h"
#include "PROlog.h"

namespace PROfit{


    int FindLocalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, int channel_index, int other_index) {
        //find local bin 
        const PROconfig::Binning &bins = inconfig.GetChannelVariableBins(channel_index, other_index);
        int ret = bins.Bin(other_value);
        if (ret == -1) {
            log<LOG_DEBUG>(L"%1% || Channel index : %2% Variable index : %3% returned under or overflow") % __func__ % channel_index % other_index;
        }
        return ret;
    }

    int FindGlobalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, int subchannel_index, int other_index) {
        int global_bin_start = inconfig.GetGlobalVariableBinStart(subchannel_index, other_index);
        int channel_index = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(subchannel_index);
        if (inconfig.GetChannelVariableBins(channel_index, other_index).NBins() == 0) {
            log<LOG_ERROR>(L"%1% || Subchannel %2% does not have other bins") % __func__ % subchannel_index;
            log<LOG_ERROR>(L"%1% || Return global bin of -1") % __func__ ;
            return -1;
        }
        int local_bin = FindLocalVariableBin(inconfig, other_value, channel_index, other_index);
        return local_bin == -1 ? -1 : global_bin_start + local_bin;
    }


    int FindGlobalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, const std::string& subchannel_fullname, int other_index) {
        int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
        return FindGlobalVariableBin(inconfig, other_value, subchannel_index, other_index);
    }


    int FindSubchannelIndexFromGlobalBin(const PROconfig &inconfig, int global_bin, int var_index ){
        return inconfig.GetSubchannelIndexFromVariableGlobalBin(global_bin,var_index);
    }

    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix){
        Eigen::MatrixXf variable_collapsing_matrix = inconfig.GetCollapsingMatrix();
        int num_bin_before_collapse = variable_collapsing_matrix.rows();
        if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
            log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        //log<LOG_DEBUG>(L"%1% || CT  %2% x %3%. Full matrix: %4% x %5% ") % __func__ % variable_collapsing_matrices[config.i_prime].transpose().rows() %  variable_collapsing_matrices[config.i_prime].transpose().cols() % full_matrix.rows() % full_matrix.cols();
        Eigen::MatrixXf result_matrix   = variable_collapsing_matrix.transpose()*full_matrix*variable_collapsing_matrix;
        return result_matrix;
    }

    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector){
        Eigen::MatrixXf variable_collapsing_matrix = inconfig.GetCollapsingMatrix();
        if(full_vector.size() != variable_collapsing_matrix.rows()){
            log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % variable_collapsing_matrix.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf result_vector = variable_collapsing_matrix.transpose() * full_vector;
        return result_vector;
    }

    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix, int other_index){
        Eigen::MatrixXf variable_collapsing_matrix = inconfig.GetCollapsingMatrix(other_index);
        int num_bin_before_collapse = variable_collapsing_matrix.rows();
        if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
            log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        //log<LOG_DEBUG>(L"%1% || CT  %2% x %3%. Full matrix: %4% x %5% ") % __func__ % variable_collapsing_matrix.transpose().rows() %  variable_collapsing_matrices[config.i_prime].transpose().cols() % full_matrix.rows() % full_matrix.cols();
        Eigen::MatrixXf result_matrix   = variable_collapsing_matrix.transpose()*full_matrix*variable_collapsing_matrix;
        return result_matrix;
    }

    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector, int other_index){
        Eigen::MatrixXf variable_collapsing_matrix = inconfig.GetCollapsingMatrix(other_index);
        if(full_vector.size() != variable_collapsing_matrix.rows()){
            log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % variable_collapsing_matrix.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf result_vector = variable_collapsing_matrix.transpose() * full_vector;
        return result_vector;
    }
    void PrintVariableInfo(const PROconfig &inconfig){

        int global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){

                        std::string temp_name  = inconfig.m_mode_names[im] +"_" +inconfig.m_detector_names[id]+"_"+inconfig.m_channel_names[ic]+"_"+inconfig.m_subchannel_names[ic][sc];
                        //global subchannel index is the main marker/identifier.
                        int global_subchannel_index1 = inconfig.GetSubchannelIndex(temp_name);
                        int local_channel_index1 = inconfig.GetLocalChannelIndexFromGlobalSubchannelIndex(global_subchannel_index1);
                        int local_channel_index3 = inconfig.GetLocalChannelIndexFromGlobalChannelIndex(global_channel_index);
                        int local_channel_index2 = ic;

                        log<LOG_INFO>(L"%1% || im:id %2%:%3%  %4%  || local_channel_index (%5%,%6%,%7%) global_channel_index (%8%) || local_subchannel_index %9% global (%10%)") % __func__ % im % id % temp_name.c_str() %  local_channel_index1 % local_channel_index2 % local_channel_index3 % global_channel_index % sc % global_subchannel_index1 ;

                        for(size_t io = 0; io < inconfig.m_num_variables; ++io) {

                            int nstart = inconfig.GetGlobalVariableBinStart(global_subchannel_index1,io);
                            int nbin = inconfig.GetChannelVariableBins(local_channel_index1,io).NBins();
                            std::vector<float> edges = inconfig.GetChannelVariableBins(local_channel_index1,io).Edges();
                            log<LOG_INFO>(L"%1% ||  ---  Vartiable %2%  || nstart %3% nbin %4% ") % __func__ % io %nstart % nbin ;
                            log<LOG_INFO>(L"%1% ||  --- -- Edges %2% ") % __func__ % edges ;
                        }
                    }
                global_channel_index++;
                }
            }
        }
        return;
    }


    
    Eigen::MatrixXf ComputeSquareRootCovariance(
            const Eigen::MatrixXf& covariance,
            bool use_ldlt,  [[maybe_unused]] float svd_tol  ) {

        if (use_ldlt) {
            // --- LDLT decomposition (faster for symmetric matrices) ---
            Eigen::LDLT<Eigen::MatrixXf> ldlt(covariance);
            if (ldlt.info() != Eigen::Success) {
                throw std::runtime_error("LDLT decomposition failed.");
            }

            // L = P^T * L_p * D^{1/2}
             Eigen::MatrixXf Lp = ldlt.matrixL();
             Eigen::VectorXf D_sqrt = ldlt.vectorD().array().sqrt();
             Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(ldlt.transpositionsP());
             return  P.transpose() * Lp * D_sqrt.asDiagonal();
        } else {

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& U = svd.matrixU();
        const auto& S = svd.singularValues();
        Eigen::VectorXf S_sqrt = S.array().sqrt();

        //float tol = 1e-8f * S.maxCoeff(); // Some cutoff? is this value impactful on out matricies? need to test
        //std::vector<int> keep;
        //for (int i = 0; i < S.size(); ++i) {
        //    if (S(i) > tol) keep.push_back(i);
        //}

        //if (keep.empty()) {
        //    log<LOG_ERROR>(L"%1% | All singular values are below tolerance, cannot sample. Blarg.") % __func__;
        //    exit(EXIT_FAILURE);
        //}
        //going to keep only the singular values that give meaningful variance
        //Eigen::MatrixXf fallback_sampler = Eigen::MatrixXf::Zero(covariance.rows(), covariance.cols());
        //for (size_t i = 0; i < keep.size(); ++i) {
        //        fallback_sampler.col(i) = U.col(keep[i]) * std::sqrt(S(keep[i]));
        //}
            return U*S_sqrt.asDiagonal();
        }
    }


    std::string getIcon(){

        std::string icon = R"(
            ....                                                                                                  
            .=+=-..                                            .:                                                 
            .:=++=-.                              .-=+*****+=-+@=.   :+##-  :#%=                                 
            .-+++=-.               .::.      .*#*=--:::--+#@@@=  :#=*@%. :@@=                                 
            .:=+++=-.            :++=-.    ...         :**=-.  .. %@#. +@+                                  
            .:-++++=:.        .=++++=-.                        :@@= =@+                                   
            .-=++++=:.     -++++++++-.                      =@@=*%-                                    
            :=+-. .-+= .:=+++++=:. .+++++++++++-.             .--.   .*%#=.                                     
            .=+=%%:  +@*.   .-++++++==+++++++++++++-.           -+++:                                             
            .. :%%: .##.      .:=++++++++++:.-+++++++-.        :+++++-.                                           
            *@* .*#.      .:..:=+++++++-   .-+++++++-.     .+++++++=:                    .::.                  
            .#@=:*+.       :*=   .-++++=.     .-+++++++-.  .=+++++++++-.                  .+*#*-.               
            =##+-         :*=     .:==.        .-=++++++-.-+++++++++++=.                    .-*%=              
            .:=+. ..        .:.-*+:::.                .:=++++++++++++=+++++++:                      -@*.            
            :**%%*.          =*+++++++.       .=-.       .-++++++++++-.:=++++++-.                     :@+            
            ..*+:*:          =*+++++++.       :++.         .-=++++++=.  .-++++++=:                     +@:           
            -%.             =*+**++++.    .::-++:::.        .:+++++.     :+++++++-.                 .+=@==.         
            +*              .::::++++.    -++++++++:       .. .-=+:       .=++++++=.                .+@@@*.         
            +#          :======: -**+.    -++++++++:      .=+:  ..         .:+++++++-.                :=:           
            :%:        .-******- .-=+.    -++++++++:       +*:               .=++++++=.                :-.   .-:.   
            +#.   .:-==++++++++==-:.     -++++++++:   .---++=--:      .::.   .-+++++++:            .-##@@:  *@@.   
            .-: :=+*****+++++++***++-.   -++++++++:   :***++***-      .++:     :=++++++-.          .=::@@-  *@+.   
            .=+*+++*+++++++++*++++*+:  -++++++++:   :+++++++*-   ....=+:...   .-++++++=:     ...    +@@. =@+     
            .=*+++++-..:*++++.-+*+++*+. -++++++++:   :+++++++*-   .===+++=+:     :+++++++-.  .=+:   .%@*.+@=      
            .+++++++.  :*++++. .+++++=: -++++++++:   :+++++++*-   .++++++++:      .=++++++=..=++=.  .#@%##:       
            .+++++++=..:*++++.  .....   -++++++++:   :+++++++*-   .++++++++:       .:+++++++=++++.   .-=:         
            .-*+++++*++++++++.          -++++++++:   :+++++++*-   .++++++++:     ....:+++++++++++-                
            .-+**++++++++++++==-:.     -++++++++:   :+++++++*-   .++++++++:  .=====++++++++++++++.               
            .-=++***+++++++***++-.   :===++===.   :+++++++*-   .++++++++:  .:-+++++++++++++++++-               
            ..:--=+++++**++++*+-.    :+=.      :+++++++*-   .++++++++:     .:=++++++++++++++=.              
            .     :*++++-=++++++*-    :+=.      :++++++++-   .:::++-::.    .-: .:=++++++++++++:              
            .=====+-    :*++++..:+++++++.   :++.      ....+*:...      .=+.    ...:*+... .:=+++++++++=.             
            .+*****+:   :*++++.  =*+++++.   .+=.         .+*:         .++:    .+++++++=    .-=+++++++:             
            :+++++*+=:.:*++++.:=+++++*-    ...          .+*:         .-=.    .++++++*=      ..-=++++-             
            :+**+++**+++++++++*+++*+-.                 .++:                 .+**++**=         .:-=++.            
            .-=+*****++++++****++=.                   ....                 .::-*+-::            .::.            
            ..:-==++++++++==-:.                                             :*+                               
            :******:.                                                 .-:.                              
            .======.                                                                                    
            .::::::::::.      .::::::::::..           .:--=--:..           .-+*####=  .--:                       
            =@@@@@@@@@@@%*-.  =@@@@@@@@@@@%*-.     .=#%@@@@@@@@%*-.       -%@@@@@@@= :@@@@-   .----.             
            =@@@@#**##@@@@@#. =@@@@#***#%@@@@*.  .+@@@@@#*++*%@@@@#:      #@@@*..... .#%%#:   .%%%%:             
            =@@@@:    .+@@@@+ =@@@@-    .=@@@@+ .*@@@@+.     .:#@@@@-  -==%@@@*====. .====. -==@@@@+===:         
            =@@@@:     -@@@@* =@@@@-     -@@@@* =@@@@+  .....  .%@@@%. %@@@@@@@@@@@- :@@@@. %@@@@@@@@@@=         
            =@@@@-::::=%@@@@: =@@@@+--==*@@@@%: #@@@@: ........ +@@@@- ---%@@@*----. :@@@@. --=@@@@*---.         
            =@@@@@@@@@@@@@%-  =@@@@@@@@@@@@#+.  #@@@@. ........ =@@@@-    #@@@=      :@@@@.   .@@@@-             
            =@@@@%%%###*+-.   =@@@@*++#@@@%:    +@@@@=  ...... .#@@@@.    #@@@=      :@@@@.   .@@@@=             
            =@@@@:            =@@@@-  .*@@@%-   .#@@@@=.      .*@@@@=     #@@@=      :@@@@.   .@@@@-             
            =@@@@:            =@@@@-   .*@@@@=   .*@@@@%*===+*@@@@@=      #@@@=      :@@@@.   .%@@@#==+:         
            =@@@@:            =@@@@-     +@@@@+.   -*%@@@@@@@@@@%+:       %@@@=      :@@@@.    =%@@@@@@*         
            :===-.            :====.      -====.     .:-=++++=-:.         -===:      .===-.     .-=++=-:         
            )";                                                                                                                                                                                    


            return icon;
    };
};
