/**
 * @file PROlog.h
 * @brief Logging infrastructure for the PROfit framework.
 * @author PROfit Collaboration
 *
 * @details Provides a lightweight, Boost.Format-based logging system with separate
 * verbosity levels for console output and optional file logging.  Messages are
 * automatically suppressed after 1000 repeated occurrences to avoid log flooding.
 *
 * Usage:
 * @code
 *   log<LOG_INFO>(L"%1% || Hello from %2%") % __func__ % "PROfit";
 * @endcode
 *
 * Verbosity levels (in increasing detail): LOG_CRITICAL, LOG_ERROR, LOG_WARNING,
 * LOG_INFO, LOG_DEBUG.
 */
#ifndef PROLOG_H_
#define PROLOG_H_

#include <sstream>
#include <boost/format.hpp>
#include <iostream>
#include <iomanip>
#include <exception>
#include <vector>
#include <fstream>
#include <unordered_map>

#include <Eigen/Eigen>

using namespace std;

/**
 * @brief Log severity levels; higher numeric value = more verbose.
 */
enum log_level_t {
    LOG_CRITICAL = 0, ///< Critical errors that will cause abnormal programme termination.
    LOG_ERROR    = 1, ///< Non-fatal errors that indicate incorrect behaviour.
    LOG_WARNING  = 2, ///< Warnings about potentially incorrect configuration or behaviour.
    LOG_INFO     = 3, ///< Informational messages describing normal operation.
    LOG_DEBUG    = 4  ///< Detailed debugging output.
};

extern log_level_t GLOBAL_LEVEL; ///< Global console verbosity level; messages at or below this level are printed.
extern log_level_t FILE_LEVEL;   ///< File verbosity level; messages at or below this level are written to the log file.
extern std::wostream *OSTREAM;   ///< Wide output stream for console logging (default: std::wcout).

extern std::wofstream LOG_FILE_STREAM; ///< Wide file stream for file logging.
extern bool LOGGING_TO_FILE;           ///< True when file logging is active.

namespace log_impl {

    /**
     * @brief Enable logging to a file in addition to console output.
     * @param filename        Path of the log file to create or overwrite.
     * @param file_verbosity  Verbosity for file output; if unset, inherits GLOBAL_LEVEL.
     */
    inline void EnableFileLogging(const std::string& filename, log_level_t file_verbosity = static_cast<log_level_t>(-1)) {
        if(LOG_FILE_STREAM.is_open()) {
            LOG_FILE_STREAM.close();
        }
        LOG_FILE_STREAM.open(filename);
        LOGGING_TO_FILE = LOG_FILE_STREAM.is_open();
        
        // If file_verbosity is not explicitly set (-1), use GLOBAL_LEVEL
        if(file_verbosity == static_cast<log_level_t>(-1)) {
            FILE_LEVEL = GLOBAL_LEVEL;
        } else {
            FILE_LEVEL = file_verbosity;
        }
        
        if(!LOGGING_TO_FILE) {
            std::wcerr << L"WARNING: Failed to open log file: " << filename.c_str() << std::endl;
        } else {
            std::wcout << L"INFO: Logging to file " << filename.c_str() 
                      << L" with verbosity " << FILE_LEVEL 
                      << L" (console verbosity: " << GLOBAL_LEVEL << L")" << std::endl;
        }
    }

    /**
     * @brief Set the file verbosity level independently of the console level.
     * @param level  New file verbosity level.
     */
    inline void SetFileVerbosity(log_level_t level) {
        FILE_LEVEL = level;
    }

    /**
     * @brief Set the console verbosity level.
     * @param level  New console verbosity level.
     */
    inline void SetConsoleVerbosity(log_level_t level) {
        GLOBAL_LEVEL = level;
    }

    /**
     * @brief Internal helper class that formats and emits a log message.
     * @details Constructed by the log<LEVEL>() helper and destroyed at the end of the
     * log statement, at which point the formatted message is written to the active stream(s).
     * Supports operator% for Boost.Format-style argument substitution, including specialisations
     * for std::vector<T> and Eigen matrix/vector types.
     */
    class formatted_log_t {
        public:
            formatted_log_t( log_level_t level, const wchar_t* msg ) : level(level), fmt(msg) {}
            ~formatted_log_t() {
                static const int SUPPRESS_AFTER = 1000;
                thread_local static std::unordered_map<std::wstring, int> msg_counts;
                std::wstring formatted_msg = boost::str(fmt);
                int& count = msg_counts[formatted_msg];
                count++;

                // Check against console verbosity
                if ( level <= GLOBAL_LEVEL ) {
                    if (count <= SUPPRESS_AFTER) {
                        *OSTREAM << level << L" " << fmt << endl;
                    } else if (count == SUPPRESS_AFTER + 1) {
                        *OSTREAM << level << L" " << fmt << L" (suppressing further cases)" << endl;
                    }
                }

                // Check against file verbosity (can be different from console)
                if(LOGGING_TO_FILE && LOG_FILE_STREAM.is_open() && level <= FILE_LEVEL) {
                    if (count <= SUPPRESS_AFTER) {
                        LOG_FILE_STREAM << level << L" " << fmt << endl;
                        LOG_FILE_STREAM.flush();
                    } else if (count == SUPPRESS_AFTER + 1) {
                        LOG_FILE_STREAM << level << L" " << fmt << L" (suppressing further cases)" << endl;
                        LOG_FILE_STREAM.flush();
                    }
                }
            }        
            
            template <typename T> 
                formatted_log_t& operator %(T value) {
                    fmt % value;
                    return *this;
                }
            template <typename T>
                formatted_log_t& operator %(const std::vector<T>& vec) {
                    std::wstringstream ss;
                    ss << L"[";
                    for (size_t i = 0; i < vec.size(); ++i) {
                        if (i != 0) ss << L", ";
                        ss << vec[i];
                    }
                    ss << L"]";
                    fmt % ss.str();
                    return *this;
                }
            template<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime>
                formatted_log_t& operator %(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>& vec) {
                    std::wstringstream ss;
                    if constexpr(ColsAtCompileTime == 1 || RowsAtCompileTime == 1) {
                        ss << L"[";
                        for (int i = 0; i < vec.size(); ++i) {
                            if (i != 0) ss << L", ";
                            ss << vec(i);
                        }
                        ss << L"]";
                    } else if constexpr(RowsAtCompileTime == -1 && ColsAtCompileTime == -1) {
                        for(int row = 0; row < vec.rows(); ++row) {
                            ss << L"\n[ ";
                            for(int col = 0; col < vec.cols(); ++col) {
                                ss << std::setw(6) << std::setprecision(3)
                                    << vec(row, col) << " ";
                            }
                            ss << L"]";
                        }
                        ss << "\n";
                    } else {
                        for(int row = 0; row < RowsAtCompileTime; ++row) {
                            ss << L"\n[ ";
                            for(int col = 0; col < ColsAtCompileTime; ++col) {
                                ss << std::setw(6) << std::setprecision(3)
                                    << vec(row, col) << " ";
                            }
                            ss << L"]";
                        }
                        ss << "\n";
                    }
                    fmt % ss.str();
                    return *this;
                }

        protected:
            log_level_t     level;
            boost::wformat      fmt;
    };

    template <>
        inline formatted_log_t& formatted_log_t::operator %(const std::vector<std::string>& vec) {
            std::wstringstream ss;
            ss << L"[";
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i != 0) ss << L", ";
                ss << vec[i].c_str();
            }
            ss << L"]";
            fmt % ss.str();
            return *this;
        }
}//namespace log_impl

/**
 * @brief Construct a log message at the given severity level.
 * @tparam level  Log severity (e.g. LOG_INFO, LOG_ERROR).
 * @param msg     Wide-character Boost.Format format string.
 * @return A formatted_log_t object; use operator% to substitute arguments.
 * @details The message is emitted when the returned object is destroyed (end of statement).
 * Example: log<LOG_INFO>(L"%1% || value = %2%") % __func__ % myValue;
 */
template <log_level_t level>
log_impl::formatted_log_t log(const wchar_t* msg) {
    return log_impl::formatted_log_t( level, msg );
}

#endif
