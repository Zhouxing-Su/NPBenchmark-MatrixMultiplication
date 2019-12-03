#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

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
    Length bestValue = input.naiveMultiplicationNum();
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
        ColorConflictError = 0x2
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::ColorConflictError) { Log(LogSwitch::Checker) << "ColorConflictError." << endl; }
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

    //status = optimizeBoolDecisionModel(sln);
    //status = optimizeRelaxedBoolDecisionModel(sln);
    //status = optimizeIntegerDecisionModel(sln);
    //status = optimizeLocalSearch(sln);
    //status = optimizeTabuSearch(sln);

    sln.multiplicationNum = input.refmultiplicationnum(); // record obj.

    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}

bool Solver::optimizeBoolDecisionModel(Solution &sln) {
    using Dvar = MpSolver::DecisionVar;

    int rowNum = input.rownuma();
    int colNum = input.colnumb();
    int numRC = input.numrcab();
    int mulNum = input.refmultiplicationnum(); // number of intermediate matrices and the number of multiplication.
    auto getRow = [&](int id) { return id / colNum; };
    auto getCol = [&](int id) { return id % colNum; };
    auto getId = [&](int row, int col) { return (row * colNum) + col; };

    auto &intermediates(*sln.mutable_intermediates());
    auto &exprs(*sln.mutable_exprs());
    
    MpSolver mp;

    // add decision variables.
    Arr<Arr2D<Dvar>> rPos(mulNum, Arr2D<Dvar>(rowNum, colNum));
    Arr<Arr2D<Dvar>> rNeg(mulNum, Arr2D<Dvar>(rowNum, colNum));
    Arr<Arr2D<Dvar>> pPos(mulNum, Arr2D<Dvar>(rowNum, numRC));
    Arr<Arr2D<Dvar>> pNeg(mulNum, Arr2D<Dvar>(rowNum, numRC));
    Arr<Arr2D<Dvar>> qPos(mulNum, Arr2D<Dvar>(numRC, colNum));
    Arr<Arr2D<Dvar>> qNeg(mulNum, Arr2D<Dvar>(numRC, colNum));
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> e(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> y(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> z(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> xPos(mulNum);
    Arr<Arr2D<Arr2D<Arr2D<Dvar>>>> xNeg(mulNum);
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                rPos[mulNum][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                rNeg[mulNum][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < numRC; ++j) {
                pPos[mulNum][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                pNeg[mulNum][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < numRC; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                qPos[mulNum][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                qNeg[mulNum][i][j] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        e[v].init(rowNum, rowNum);
        y[v].init(rowNum, rowNum);
        z[v].init(rowNum, rowNum);
        xPos[v].init(rowNum, rowNum);
        xNeg[v].init(rowNum, rowNum);
        for (ID i = 0; i < rowNum; ++i) {
            for (ID ii = 0; ii < rowNum; ++ii) {
                e[v][i][ii].init(colNum, colNum);
                y[v][i][ii].init(colNum, colNum);
                z[v][i][ii].init(colNum, colNum);
                xPos[v][i][ii].init(colNum, colNum);
                xNeg[v][i][ii].init(colNum, colNum);
                for (ID j = 0; j < colNum; ++j) {
                    for (ID jj = 0; jj < colNum; ++jj) {
                        e[v][i][ii][j][jj].init(numRC, numRC);
                        y[v][i][ii][j][jj].init(numRC, numRC);
                        z[v][i][ii][j][jj].init(numRC, numRC);
                        xPos[v][i][ii][j][jj].init(numRC, numRC);
                        xNeg[v][i][ii][j][jj].init(numRC, numRC);
                        for (ID k = 0; k < numRC; ++k) {
                            for (ID kk = 0; kk < numRC; ++kk) {
                                e[mulNum][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                y[mulNum][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                z[mulNum][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                xPos[mulNum][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
                                xNeg[mulNum][i][ii][j][jj][k][kk] = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
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
                mp.addConstraint(rPos[mulNum][i][j] + rNeg[mulNum][i][j] <= 1);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < rowNum; ++i) {
            for (ID j = 0; j < numRC; ++j) {
                mp.addConstraint(pPos[mulNum][i][j] + pNeg[mulNum][i][j] <= 1);
            }
        }
    }
    for (ID v = 0; v < mulNum; ++v) {
        for (ID i = 0; i < numRC; ++i) {
            for (ID j = 0; j < colNum; ++j) {
                mp.addConstraint(qPos[mulNum][i][j] + qNeg[mulNum][i][j] <= 1);
            }
        }
    }

    // 

    // solve model.
    mp.setOutput(true);
    //mp.setMaxThread(1);
    mp.setTimeLimitInSecond(1800);
    //mp.setMipFocus(MpSolver::MipFocusMode::ImproveFeasibleSolution);

    // record decision.
    //if (mp.optimize()) {
    //    for (ID n = 0; n < nodeNum; ++n) {
    //        for (ID c = 0; c < input.colornum(); ++c) {
    //            if (mp.isTrue(isColor.at(n, c))) { nodeColors[n] = c; break; }
    //        }
    //    }
    //    return true;
    //}

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
