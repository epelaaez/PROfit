#include "PROtocall.h"
#include "PROlog.h"

namespace PROfit{


    int FindLocalVariableBin(const PROconfig &inconfig, float other_value, int channel_index, int other_index) {
        //find local bin 
        const std::vector<float>& bin_edges = inconfig.GetChannelVariableBinEdges(channel_index, other_index);
        auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), other_value);

        //over/under-flow, don't care for now
        if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
            log<LOG_DEBUG>(L"%1% || True value: %2% is in underflow or overflow bins, return bin of -1") % __func__ % other_value;
            log<LOG_DEBUG>(L"%1% || Channel %2% has bin lower edge: %3% and bin upper edge: %4%") % __func__ % channel_index % *bin_edges.begin() % bin_edges.back();
            return -1; 
        }
        return pos_iter - bin_edges.begin() - 1; 
    }

    int FindGlobalVariableBin(const PROconfig &inconfig, float other_value, int subchannel_index, int other_index) {
        int global_bin_start = inconfig.GetGlobalVariableBinStart(subchannel_index, other_index);
        int channel_index = inconfig.GetChannelIndex(subchannel_index);
        if(inconfig.GetChannelNVariableBins(channel_index, other_index) == 0){
            log<LOG_ERROR>(L"%1% || Subchannel %2% does not have other bins") % __func__ % subchannel_index;
            log<LOG_ERROR>(L"%1% || Return global bin of -1") % __func__ ;
            return -1;
        }
        int local_bin = FindLocalVariableBin(inconfig, other_value, channel_index, other_index);
        return local_bin == -1 ? -1 : global_bin_start + local_bin;
    }


    int FindGlobalVariableBin(const PROconfig &inconfig, float other_value, const std::string& subchannel_fullname, int other_index) {
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
