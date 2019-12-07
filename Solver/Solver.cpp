#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <array>

#include <cmath>

#include "MpSolver.h"


using namespace std;


namespace szx {

#pragma region Solver::Cli
int Solver::Cli::run(int argc, char * argv[]) {
    Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
    Set<String> switchSet;
    Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
        { InstancePathOption(), nullptr },
        { SolutionPathOption(), nullptr },
        { RandSeedOption(), nullptr },
        { TimeoutOption(), nullptr },
        { MaxIterOption(), nullptr },
        { JobNumOption(), nullptr },
        { RunIdOption(), nullptr },
        { EnvironmentPathOption(), nullptr },
        { ConfigPathOption(), nullptr },
        { LogPathOption(), nullptr }
    });

    for (int i = 1; i < argc; ++i) { // skip executable name.
        auto mapIter = optionMap.find(argv[i]);
        if (mapIter != optionMap.end()) { // option argument.
            mapIter->second = argv[++i];
        } else { // switch argument.
            switchSet.insert(argv[i]);
        }
    }

    Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
    if (switchSet.find(HelpSwitch()) != switchSet.end()) {
        cout << HelpInfo() << endl;
    }

    if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
        cout << AuthorName() << endl;
    }

    Solver::Environment env;
    env.load(optionMap);
    if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

    Solver::Configuration cfg;
    cfg.load(env.cfgPath);

    Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
    Problem::Input input;
    if (!input.load(env.instPath)) { return -1; }

    Solver solver(input, env, cfg);
    solver.solve();

    pb::Submission submission;
    submission.set_thread(to_string(env.jobNum));
    submission.set_instance(env.friendlyInstName());
    submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

    solver.output.save(env.slnPath, submission);
    #if SZX_DEBUG
    solver.output.save(env.solutionPathWithTime(), submission);
    solver.record();
    #endif // SZX_DEBUG

    return 0;
}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
void Solver::Environment::load(const Map<String, char*> &optionMap) {
    char *str;

    str = optionMap.at(Cli::EnvironmentPathOption());
    if (str != nullptr) { loadWithoutCalibrate(str); }

    str = optionMap.at(Cli::InstancePathOption());
    if (str != nullptr) { instPath = str; }

    str = optionMap.at(Cli::SolutionPathOption());
    if (str != nullptr) { slnPath = str; }

    str = optionMap.at(Cli::RandSeedOption());
    if (str != nullptr) { randSeed = atoi(str); }

    str = optionMap.at(Cli::TimeoutOption());
    if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

    str = optionMap.at(Cli::MaxIterOption());
    if (str != nullptr) { maxIter = atoi(str); }

    str = optionMap.at(Cli::JobNumOption());
    if (str != nullptr) { jobNum = atoi(str); }

    str = optionMap.at(Cli::RunIdOption());
    if (str != nullptr) { rid = str; }

    str = optionMap.at(Cli::ConfigPathOption());
    if (str != nullptr) { cfgPath = str; }

    str = optionMap.at(Cli::LogPathOption());
    if (str != nullptr) { logPath = str; }

    calibrate();
}

void Solver::Environment::load(const String &filePath) {
    loadWithoutCalibrate(filePath);
    calibrate();
}

void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
    // EXTEND[szx][8]: load environment from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Environment::save(const String &filePath) const {
    // EXTEND[szx][8]: save environment to file.
}
void Solver::Environment::calibrate() {
    // adjust thread number.
    int threadNum = thread::hardware_concurrency();
    if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

    // adjust timeout.
    msTimeout -= Environment::SaveSolutionTimeInMillisecond;
}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
void Solver::Configuration::load(const String &filePath) {
    // EXTEND[szx][5]: load configuration from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Configuration::save(const String &filePath) const {
    // EXTEND[szx][5]: save configuration to file.
}
#pragma endregion Solver::Configuration

#pragma region Solver
bool Solver::solve() {
    init();

    int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
    cfg.threadNumPerWorker = env.jobNum / workerNum;
    List<Solution> solutions(workerNum, Solution(this));
    List<bool> success(workerNum);

    Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
    List<thread> threadList;
    threadList.reserve(workerNum);
    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Length bestValue = input.naiveMultiplicationNum() + 1;
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].multiplicationNum << endl;
        if (solutions[i].multiplicationNum >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].multiplicationNum;
    }

    env.rid = to_string(bestIndex);
    if (bestIndex < 0) { return false; }
    output = solutions[bestIndex];
    return true;
}

void Solver::record() const {
    #if SZX_DEBUG
    int generation = 0;

    ostringstream log;

    System::MemoryUsage mu = System::peakMemoryUsage();

    Length obj = output.multiplicationNum;
    Length checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ","
        << obj << ","
        << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    // EXTEND[szx][5]: save results in plain text.
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Color,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
    }
    logFile << log.str();
    logFile.close();
    #endif // SZX_DEBUG
}

bool Solver::check(Length &checkerObj) const {
    #if SZX_DEBUG
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        WorsePerformanceError = 0x2,
        WorstPerformanceError = 0x4,
        WrongResultError = 0x8
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::WorsePerformanceError) { Log(LogSwitch::Checker) << "WorsePerformanceError." << endl; }
    if (checkerObj & CheckerFlag::WorstPerformanceError) { Log(LogSwitch::Checker) << "WorstPerformanceError." << endl; }
    if (checkerObj & CheckerFlag::WrongResultError) { Log(LogSwitch::Checker) << "WrongResultError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {

}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

    // reset solution state.
    bool status = false;

    //status = optimizePlainModel(sln);
    status = optimizePatternPickingModel(sln);
    //status = optimizeIntegerDecisionModel(sln);
    //status = optimizeLocalSearch(sln);
    //status = optimizeTabuSearch(sln);

    sln.multiplicationNum = input.refmultiplicationnum(); // record obj.

    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}

bool Solver::optimizePlainModel(Solution &sln) {
    using Dvar = MpSolver::DecisionVar;
    using Expr = MpSolver::LinearExpr;

    constexpr bool ShouldRelax = true;
    constexpr bool ShouldAddAtLeastOneNegativeTermCut = false;

    ID rowNum = input.rownuma();
    ID colNum = input.colnumb();
    ID numRC = input.numrcab();
    ID mulNum = input.refmultiplicationnum(); // number of intermediate matrices and the number of multiplication.
    auto getRow = [&](ID id) { return id / colNum; };
    auto getCol = [&](ID id) { return id % colNum; };
    auto getId = [&](ID row, ID col) { return (row * colNum) + col; };
    
    MpSolver mp;

    // add decision variables.
    Arr<Arr2D<Dvar>> rPos(mulNum, Arr2D<Dvar>(rowNum, colNum));
    Arr<Arr2D<Dvar>> rNeg(mulNum, Arr2D<Dvar>(rowNum, colNum));
    Arr<Arr2D<Dvar>> pPos(mulNum, Arr2D<Dvar>(rowNum, numRC));
    Arr<Arr2D<Dvar>> pNeg(mulNum, Arr2D<Dvar>(rowNum, numRC));
    Arr<Arr2D<Dvar>> qPos(mulNum, Arr2D<Dvar>(numRC, colNum));
    Arr<Arr2D<Dvar>> qNeg(mulNum, Arr2D<Dvar>(numRC, colNum));
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> o(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> y(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> z(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> xPos(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> xNeg(mulNum);
    Arr2D<Arr2D<Arr2D<Dvar>>> slacks(rowNum, rowNum);
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                rPos[v][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                rNeg[v][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < numRC; ++j) {
                pPos[v][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                pNeg[v][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < numRC; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                qPos[v][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                qNeg[v][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        o[v].init(rowNum, rowNum);
        y[v].init(rowNum, rowNum);
        z[v].init(rowNum, rowNum);
        xPos[v].init(rowNum, rowNum);
        xNeg[v].init(rowNum, rowNum);
        for (ID i = 0; i < rowNum; ++i) {
            for (ID ii = 0; ii < rowNum; ++ii) {
                o[v][i][ii].init(colNum, colNum);
                y[v][i][ii].init(colNum, colNum);
                z[v][i][ii].init(colNum, colNum);
                xPos[v][i][ii].init(colNum, colNum);
                xNeg[v][i][ii].init(colNum, colNum);
                for (ID j = 0; j < colNum; ++j) {
                    for (ID jj = 0; jj < colNum; ++jj) {
                        o[v][i][ii][j][jj].init(numRC, numRC);
                        y[v][i][ii][j][jj].init(numRC, numRC);
                        z[v][i][ii][j][jj].init(numRC, numRC);
                        xPos[v][i][ii][j][jj].init(numRC, numRC);
                        xNeg[v][i][ii][j][jj].init(numRC, numRC);
                        for (ID k = 0; k < numRC; ++k) {
                            for (ID kk = 0; kk < numRC; ++kk) {
                                o[v][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                y[v][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                z[v][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                xPos[v][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                xNeg[v][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                            }
                        }
                    }
                }
            }
        }
    }
    if (ShouldRelax) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID ii = 0; ii < rowNum; ++ii) {
                slacks[i][ii].init(colNum, colNum);
                for (ID j = 0; j < colNum; ++j) {
                    for (ID jj = 0; jj < colNum; ++jj) {
                        slacks[i][ii][j][jj].init(numRC, numRC);
                        for (ID k = 0; k < numRC; ++k) {
                            for (ID kk = 0; kk < numRC; ++kk) {
                                slacks[i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Real, 0, mulNum, 1);
                            }
                        }
                    }
                }
            }
        }
    }

    // add constraints.
    // single value.
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                mp.addConstraint(rPos[v][i][j] + rNeg[v][i][j] <= 1);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < numRC; ++j) {
                mp.addConstraint(pPos[v][i][j] + pNeg[v][i][j] <= 1);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < numRC; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                mp.addConstraint(qPos[v][i][j] + qNeg[v][i][j] <= 1);
            }
        }
    }

    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID ii = 0; ii < rowNum; ++ii) {
                for (ID j = 0; j < colNum; ++j) {
                    for (ID jj = 0; jj < colNum; ++jj) {
                        for (ID k = 0; k < numRC; ++k) {
                            for (ID kk = 0; kk < numRC; ++kk) {
                                // non-positive or non-negative.
                                mp.addConstraint(rNeg[v][i][j] + pNeg[v][ii][k] + qNeg[v][kk][jj] == 2 * y[v][i][ii][j][jj][k][kk] + o[v][i][ii][j][jj][k][kk]);

                                // non-zero.
                                Expr sum = rPos[v][i][j] + rNeg[v][i][j] + pPos[v][ii][k] + pNeg[v][ii][k] + qPos[v][kk][jj] + qNeg[v][kk][jj];
                                mp.addConstraint(3 * z[v][i][ii][j][jj][k][kk] <= sum);
                                mp.addConstraint(sum <= 2 + z[v][i][ii][j][jj][k][kk]);

                                // negative one.
                                mp.addConstraint(2 * xNeg[v][i][ii][j][jj][k][kk] <= z[v][i][ii][j][jj][k][kk] + o[v][i][ii][j][jj][k][kk]);
                                mp.addConstraint(z[v][i][ii][j][jj][k][kk] + o[v][i][ii][j][jj][k][kk] <= 1 + xNeg[v][i][ii][j][jj][k][kk]);
                                // positive one.
                                mp.addConstraint(2 * xPos[v][i][ii][j][jj][k][kk] <= z[v][i][ii][j][jj][k][kk] + 1 - o[v][i][ii][j][jj][k][kk]);
                                mp.addConstraint(z[v][i][ii][j][jj][k][kk] + 1 - o[v][i][ii][j][jj][k][kk] <= 1 + xPos[v][i][ii][j][jj][k][kk]);
                            }
                        }
                    }
                }
            }
        }
    }

    // matched terms.
    for (ID i = 0; i < rowNum; ++i) {
        for (ID ii = 0; ii < rowNum; ++ii) {
            for (ID j = 0; j < colNum; ++j) {
                for (ID jj = 0; jj < colNum; ++jj) {
                    for (ID k = 0; k < numRC; ++k) {
                        for (ID kk = 0; kk < numRC; ++kk) {
                            Expr sum;
                            for (ID v = 0; v < mulNum; ++v) {
                                sum += xPos[v][i][ii][j][jj][k][kk];
                                sum -= xNeg[v][i][ii][j][jj][k][kk];
                            }
                            bool termExists = ((i == ii) && (j == jj) && (k == kk));
                            if (ShouldRelax) {
                                mp.addConstraint(sum >= termExists - slacks[i][ii][j][jj][k][kk]);
                                mp.addConstraint(sum <= termExists + slacks[i][ii][j][jj][k][kk]);
                            } else {
                                mp.addConstraint(sum == termExists);
                            }
                        }
                    }
                }
            }
        }
    }

    #if SZX_VERIFY_2X2_MODEL
    double p[7][2][2] = {
        {{  1,  0 },
         {  0,  1 }},
        {{  0,  0 },
         {  1,  1 }},
        {{  1,  0 },
         {  0,  0 }},
        {{  0,  0 },
         {  0,  1 }},
        {{  1,  1 },
         {  0,  0 }},
        {{ -1,  0 },
         {  1,  0 }},
        {{  0,  1 },
         {  0, -1 }},
    };
    double q[7][2][2] = {
        {{  1,  0 },
         {  0,  1 }},
        {{  1,  0 },
         {  0,  0 }},
        {{  0,  1 },
         {  0, -1 }},
        {{ -1,  0 },
         {  1,  0 }},
        {{  0,  0 },
         {  0,  1 }},
        {{  1,  1 },
         {  0,  0 }},
        {{  0,  0 },
         {  1,  1 }},
    };
    double r[7][2][2] = {
        {{  1,  0 },
         {  0,  1 }},
        {{  0,  0 },
         {  1, -1 }},
        {{  0,  1 },
         {  0,  1 }},
        {{  1,  0 },
         {  1,  0 }},
        {{ -1,  1 },
         {  0,  0 }},
        {{  0,  0 },
         {  0,  1 }},
        {{  1,  0 },
         {  0,  0 }},
    };
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                mp.addConstraint(pPos[v][i][j] == ((p[v][i][j] > 0) ? 1 : 0));
                mp.addConstraint(pNeg[v][i][j] == ((p[v][i][j] < 0) ? 1 : 0));
                mp.addConstraint(qPos[v][i][j] == ((q[v][i][j] > 0) ? 1 : 0));
                mp.addConstraint(qNeg[v][i][j] == ((q[v][i][j] < 0) ? 1 : 0));
                mp.addConstraint(rPos[v][i][j] == ((r[v][i][j] > 0) ? 1 : 0));
                mp.addConstraint(rNeg[v][i][j] == ((r[v][i][j] < 0) ? 1 : 0));
            }
        }
    }
    #endif

    // at-least-one-negative-term cut.
    if (ShouldAddAtLeastOneNegativeTermCut) {
        Expr sum;
        for (ID v = 0; v < mulNum; ++v) {
            for (ID i = 0; i < rowNum; ++i) {
                for (ID j = 0; j < colNum; ++j) {
                    sum += rNeg[v][i][j];
                }
            }
        }
        mp.addConstraint(sum >= 1);
        sum = 0;
        for (ID v = 0; v < mulNum; ++v) {
            for (ID i = 0; i < rowNum; ++i) {
                for (ID j = 0; j < numRC; ++j) {
                    sum += pNeg[v][i][j];
                }
            }
        }
        mp.addConstraint(sum >= 1);
        sum = 0;
        for (ID v = 0; v < mulNum; ++v) {
            for (ID i = 0; i < numRC; ++i) {
                for (ID j = 0; j < colNum; ++j) {
                    sum += qNeg[v][i][j];
                }
            }
        }
        mp.addConstraint(sum >= 1);
    }

    // solve model.
    mp.setOutput(true);
    //mp.setMaxThread(1);
    mp.setTimeLimitInSecond(3600);
    //mp.setMipFocus(MpSolver::MipFocusMode::ImproveFeasibleSolution);

    // record decision.
    if (mp.optimize()) {
        // init solution vector.
        auto &intermediates(*sln.mutable_intermediates());
        intermediates.Reserve(mulNum);
        for (ID v = 0; v < mulNum; ++v) { intermediates.Add(); }

        auto &exprs(*sln.mutable_exprs());
        exprs.Reserve(rowNum * colNum);
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) { exprs.Add(); }
        }

        // retrieve values.
        for (ID v = 0; v < mulNum; ++v) {
            for (ID i = 0, id = 0; i < rowNum; ++i) {
                for (ID j = 0; j < colNum; ++j, ++id) {
                    if (mp.isTrue(rPos[v][i][j])) {
                        Solution::addTerm(exprs[id], v, 1);
                    } else if (mp.isTrue(rNeg[v][i][j])) {
                        Solution::addTerm(exprs[id], v, -1);
                    } // else (r == 0).
                }
            }
            auto &aExpr(*intermediates[v].mutable_suma());
            for (ID i = 0, id = 0; i < rowNum; ++i) {
                for (ID j = 0; j < numRC; ++j, ++id) {
                    if (mp.isTrue(pPos[v][i][j])) {
                        Solution::addTerm(aExpr, id, 1);
                    } else if (mp.isTrue(pNeg[v][i][j])) {
                        Solution::addTerm(aExpr, id, -1);
                    } // else (p == 0).
                }
            }
            auto &bExpr(*intermediates[v].mutable_sumb());
            for (ID i = 0, id = 0; i < numRC; ++i) {
                for (ID j = 0; j < colNum; ++j, ++id) {
                    if (mp.isTrue(qPos[v][i][j])) {
                        Solution::addTerm(bExpr, id, 1);
                    } else if (mp.isTrue(qNeg[v][i][j])) {
                        Solution::addTerm(bExpr, id, -1);
                    } // else (q == 0).
                }
            }
        }
        return true;
    }

    return false;
}

bool Solver::optimizePatternPickingModel(Solution &sln) {
    using Dvar = MpSolver::DecisionVar;
    using Expr = MpSolver::LinearExpr;

    constexpr bool AddIntermediateMatrixNumCut = false;
    constexpr double MinR = -1;
    constexpr double MaxR = 1;
    constexpr ID DomainSizePQ = 3;
    constexpr array<float, DomainSizePQ> DomainPQ = { 0, 1, -1 };

    ID rowNum = input.rownuma();
    ID colNum = input.colnumb();
    ID numRC = input.numrcab();
    ID mulNum = input.refmultiplicationnum(); // number of intermediate matrices and the number of multiplication.

    // generate pool.
    if ((rowNum != numRC) || (numRC != colNum)) { Log(LogSwitch::Szx::Preprocess) << "[Error] square matrices only." << endl; return false; }

    ID abPatternDepth = rowNum * numRC;
    // EXTEND[szx][5]: in order to support rect matrices, distinct pools for a and b should be used.
    if (rowNum >= 5) { Log(LogSwitch::Szx::Preprocess) << "[Error] abPoolSize overflow." << endl; return false; }
    ID abPoolSize = static_cast<ID>(pow(DomainPQ.size(), abPatternDepth) - 1);
    List<Arr2D<float>> abPool;
    abPool.reserve(abPoolSize);

    Arr2D<float> abPattern(rowNum, numRC);
    abPattern.reset();

    struct StackItem {
        ID domain;
    };
    List<StackItem> enumStack(abPatternDepth, { 1 }); // skip the all-zero pattern.
    for (ID id = 3; !enumStack.empty();) {
        StackItem &si(enumStack.back());
        if (si.domain >= DomainSizePQ) { // backtrack.
            enumStack.pop_back();
            --id;
        } else {
            abPattern.at(id) = DomainPQ[si.domain];
            ++si.domain;
            if (++id < abPatternDepth) { // go deeper.
                enumStack.push_back({ 0 });
            } else { // a pattern is complete.
                abPool.push_back(abPattern);
                --id;
            }
        }
    }

    #if SZX_VERIFY_2X2_MODEL
    float pp[7][2][2] = {
        {{  1,  0 },
         {  0,  1 }},
        {{  0,  0 },
         {  1,  1 }},
        {{  1,  0 },
         {  0,  0 }},
        {{  0,  0 },
         {  0,  1 }},
        {{  1,  1 },
         {  0,  0 }},
        {{ -1,  0 },
         {  1,  0 }},
        {{  0,  1 },
         {  0, -1 }},
    };
    abPoolSize = 7;
    abPool.resize(abPoolSize);
    for (ID v = 0; v < abPoolSize; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                abPool[v][i][j] = pp[v][i][j];
            }
        }
    }
    #endif

    if (rowNum >= 4) { Log(LogSwitch::Szx::Preprocess) << "[Error] mPoolSize overflow." << endl; return false; }
    ID mPoolSize = abPoolSize * abPoolSize;
    List<array<ID, 2>> mPool; // a x b.
    mPool.reserve(mPoolSize);
    for (ID aId = 0; aId < abPoolSize; ++aId) {
        for (ID bId = 0; bId < abPoolSize; ++bId) {
            mPool.push_back({ aId, bId });
        }
    }

    MpSolver mp;

    // add decision variables.
    Arr<Dvar> rUsed(mPoolSize);
    Arr<Arr2D<Dvar>> r(mPoolSize, Arr2D<Dvar>(rowNum, colNum));
    for (ID v = 0; v < mPoolSize; ++v) {
        rUsed[v] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, !AddIntermediateMatrixNumCut);
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                r[v][i][j] = mp.addVar(MpSolver::VariableType::Real, MinR, MaxR, 0);
            }
        }
    }

    // add constraints.
    // used terms.
    for (ID v = 0; v < mPoolSize; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                mp.addConstraint(MinR * rUsed[v] <= r[v][i][j]);
                mp.addConstraint(r[v][i][j] <= MaxR * rUsed[v]);
            }
        }
    }

    // matched terms.
    for (ID i = 0; i < rowNum; ++i) {
        for (ID ii = 0; ii < rowNum; ++ii) {
            for (ID j = 0; j < colNum; ++j) {
                for (ID jj = 0; jj < colNum; ++jj) {
                    for (ID k = 0; k < numRC; ++k) {
                        for (ID kk = 0; kk < numRC; ++kk) {
                            Expr sum;
                            for (ID v = 0; v < mPoolSize; ++v) {
                                const Arr2D<float> &p = abPool[mPool[v][0]];
                                const Arr2D<float> &q = abPool[mPool[v][1]];
                                sum += p[ii][k] * q[kk][jj] * r[v][i][j];
                            }
                            bool termExists = ((i == ii) && (j == jj) && (k == kk));
                            mp.addConstraint(sum == termExists);
                        }
                    }
                }
            }
        }
    }

    // intermediate matrix number cut.
    if (AddIntermediateMatrixNumCut) {
        Expr matNum;
        for (ID v = 0; v < mPoolSize; ++v) { matNum += rUsed[v]; }
        mp.addConstraint(matNum == mulNum);
        //mp.addConstraint(matNum <= mulNum);
    }

    // solve model.
    mp.setOutput(true);
    //mp.setMaxThread(1);
    mp.setTimeLimitInSecond(3600);
    //mp.setMipFocus(MpSolver::MipFocusMode::ImproveFeasibleSolution);

    // record decision.
    if (mp.optimize()) {
        // init solution vector.
        auto &intermediates(*sln.mutable_intermediates());
        intermediates.Reserve(mulNum);
        for (ID v = 0; v < mulNum; ++v) { intermediates.Add(); }

        auto &exprs(*sln.mutable_exprs());
        exprs.Reserve(rowNum * colNum);
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) { exprs.Add(); }
        }

        // retrieve values.
        for (ID v = 0, vUsed = 0; v < mPoolSize; ++v) {
            if (!mp.isTrue(rUsed[v])) { continue; }
            for (ID i = 0, id = 0; i < rowNum; ++i) {
                for (ID j = 0; j < colNum; ++j, ++id) {
                    Solution::addTerm(exprs[id], vUsed, mp.getValue(r[v][i][j]));
                }
            }
            auto &aExpr(*intermediates[vUsed].mutable_suma());
            for (ID i = 0, id = 0; i < rowNum; ++i) {
                for (ID j = 0; j < numRC; ++j, ++id) {
                    float p = abPool[mPool[v][0]][i][j];
                    if (p != 0) { Solution::addTerm(aExpr, id, p); }
                }
            }
            auto &bExpr(*intermediates[vUsed].mutable_sumb());
            for (ID i = 0, id = 0; i < numRC; ++i) {
                for (ID j = 0; j < colNum; ++j, ++id) {
                    float q = abPool[mPool[v][1]][i][j];
                    if (q != 0) { Solution::addTerm(bExpr, id, q); }
                }
            }
            ++vUsed;
        }
        return true;
    }

    return false;
}

bool Solver::optimizeLocalSearch(Solution &sln) {
    return false;
}

bool Solver::optimizeTabuSearch(Solution &sln) {
    return false;
}
#pragma endregion Solver

}
