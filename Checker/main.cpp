#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <limits>
#include <vector>
#include <set>

#include <cmath>

#include "../Solver/PbReader.h"
#include "../Solver/MatrixMultiplication.pb.h"


using namespace std;
using namespace pb;


int main(int argc, char *argv[]) {
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        WorsePerformanceError = 0x2,
        WrongResultError = 0x4
    };

    string inputPath;
    string outputPath;

    if (argc > 1) {
        inputPath = argv[1];
    } else {
        cerr << "input path: " << flush;
        cin >> inputPath;
    }

    if (argc > 2) {
        outputPath = argv[2];
    } else {
        cerr << "output path: " << flush;
        cin >> outputPath;
    }

    pb::MatrixMultiplication::Input input;
    if (!load(inputPath, input)) { return ~CheckerFlag::IoError; }

    pb::MatrixMultiplication::Output output;
    ifstream ifs(outputPath);
    if (!ifs.is_open()) { return ~CheckerFlag::IoError; }
    string submission;
    getline(ifs, submission); // skip the first line.
    ostringstream oss;
    oss << ifs.rdbuf();
    jsonToProtobuf(oss.str(), output);

    int rowNum = input.rownuma();
    int colNum = input.colnumb();
    int numRC = input.numrcab();
    int mulNum = output.intermediates().size(); // number of intermediate matrices and the number of multiplication.
    auto getRow = [&](int id) { return id / colNum; };
    auto getCol = [&](int id) { return id % colNum; };
    auto getId = [&](int row, int col) { return (row * colNum) + col; };
    auto isPowOfTwo = [](double d) {
        int exp;
        return abs(frexp(d, &exp)) == 0.5;
    };

    if (output.exprs().size() != (rowNum * colNum)) { return ~CheckerFlag::FormatError; }
    if (mulNum >= (rowNum * input.numrcab() * colNum)) { return ~CheckerFlag::WorsePerformanceError; }
    if (mulNum > input.refmultiplicationnum()) { return ~CheckerFlag::WorsePerformanceError; }

    // check solution.
    int error = 0;
    // check constraints.
    vector<vector<double>> r(mulNum, vector<double>(rowNum * colNum, 0));
    vector<vector<double>> p(mulNum, vector<double>(rowNum * colNum, 0));
    vector<vector<double>> q(mulNum, vector<double>(rowNum * colNum, 0));
    for (auto m = 0; m < mulNum; ++m) {
        auto &med(output.intermediates(m));
        for (auto t = med.suma().terms().begin(); t != med.suma().terms().end(); ++t) {
            p[m][t->id()] += t->coef();
            if (!isPowOfTwo(t->coef())) { cerr << "[Warning] a not-power-of-two coefficient may cause false-positive error report due to floating-point arithmetic."; }
        }
        for (auto t = med.sumb().terms().begin(); t != med.sumb().terms().end(); ++t) {
            q[m][t->id()] += t->coef();
            if (!isPowOfTwo(t->coef())) { cerr << "[Warning] a not-power-of-two coefficient may cause false-positive error report due to floating-point arithmetic."; }
        }
    }
    auto e = output.exprs().begin();
    for (int i = 0; i < rowNum; ++i) {
        for (int j = 0; j < colNum; ++j, ++e) {
            for (auto t = e->terms().begin(); t != e->terms().end(); ++t) {
                r[t->id()][getId(i, j)] += t->coef();
            }
        }
    }

    for (int i = 0; i < rowNum; ++i) {
        for (int ii = 0; ii < rowNum; ++ii) {
            for (int j = 0; j < colNum; ++j) {
                for (int jj = 0; jj < colNum; ++jj) {
                    for (int k = 0; k < numRC; ++k) {
                        for (int kk = 0; kk < numRC; ++kk) {
                            double sum = 0;
                            for (int m = 0; m < mulNum; ++m) {
                                sum += r[m][getId(i, j)] * p[m][getId(ii, k)] * q[m][getId(kk, jj)];
                            }
                            double correctSum = ((i == ii) && (j == jj) && (k == kk));
                            if (sum != correctSum) { error |= CheckerFlag::WrongResultError; }
                        }
                    }
                }
            }
        }
    }
    // check objective.
    int multiplicationNum = mulNum;

    int returnCode = (error == 0) ? multiplicationNum : ~error;
    cout << ((error == 0) ? multiplicationNum : returnCode) << endl;
    return returnCode;
}
