////////////////////////////////
/// usage : 1.	data that identifies a guillotine cut problem and its solution.
/// 
/// note  : 1.	
////////////////////////////////

#ifndef SMART_SZX_MATRIX_MULTIPLICATION_PROBLEM_H
#define SMART_SZX_MATRIX_MULTIPLICATION_PROBLEM_H


#include "Config.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "Common.h"
#include "PbReader.h"
#include "MatrixMultiplication.pb.h"


namespace szx {

class Problem {
    #pragma region Type
public:
    struct Input : public pb::MatrixMultiplication::Input {
        bool load(const String &path) { return pb::load(path, *this); }

        ID naiveMultiplicationNum() const { return rownuma() * numrcab() * colnumb(); }
    };

    struct Output : public pb::MatrixMultiplication::Output {
        bool save(const String &path, pb::Submission &submission) const {
            std::ofstream ofs(path);
            if (!ofs.is_open()) { return false; }

            // TODO[0]: fill the submission information.
            submission.set_author("szx");
            submission.set_algorithm("rand");
            submission.set_cpu("Intel Core i5-7400 3.00GHz");
            submission.set_ram("16G 2400MHz");
            submission.set_language("C++");
            submission.set_compiler("VS2017");
            submission.set_os("Windows 10");
            submission.set_problem("GraphColoring");

            ofs << protobufToJson(submission, false) << std::endl << protobufToJson(*this);
            return true;
        }

        ID multiplicationNum = 0;
    };
    #pragma endregion Type

    #pragma region Constant
public:
    enum {
        MaxDimensionNum = 100,

        InvalidId = -1,
    };
    #pragma endregion Constant

    #pragma region Constructor
public:
    #pragma endregion Constructor

    #pragma region Method
public:
    #pragma endregion Method

    #pragma region Field
public:
    #pragma endregion Field
}; // Problem

}


#endif // SMART_SZX_MATRIX_MULTIPLICATION_PROBLEM_H
