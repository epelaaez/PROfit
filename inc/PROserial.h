#ifndef PROSERIEL_H
#define PROSERIEL_H


#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <fstream>
#include <Eigen/Dense>


//extrnd boost to handle read/write serialization of Eigen::MatrixXf and Eigen::VectorXF
namespace boost {
    namespace serialization {

        template <class Archive>
            void serialize(Archive& ar, Eigen::MatrixXf& mat, [[maybe_unused]] const unsigned int version) {
                int rows = mat.rows(), cols = mat.cols();
                ar & rows & cols;
                if (Archive::is_loading::value) mat.resize(rows, cols);
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        ar & mat(i, j);
            }

        template <class Archive>
            void serialize(Archive& ar, Eigen::VectorXf& vec, [[maybe_unused]] const unsigned int version) {
                int size = vec.size();
                ar & size;
                if (Archive::is_loading::value) vec.resize(size);
                for (int i = 0; i < size; ++i)
                    ar & vec(i);
            }


        // boost 4serialize std::array<T, N>
        template<class Archive, typename T, std::size_t N>
            void serialize(Archive& ar, std::array<T, N>& arr, [[maybe_unused]]  const unsigned int version) {
                 ar & boost::serialization::make_nvp("array", arr);
            }

    } 
} 

#endif

