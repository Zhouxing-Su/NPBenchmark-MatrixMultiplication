#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <array>
#include <vector>
#include <algorithm>
#include <random>

#include <cstring>

#include "Simulator.h"
#include "ThreadPool.h"


using namespace std;


namespace szx {

// EXTEND[szx][5]: read it from InstanceList.txt.
static const vector<String> instList({
    "square.2x2x2m7",
    "square.2x2x2m6",
    "square.3x3x3m23",
    "square.3x3x3m22",
    "square.3x3x3m21",
});

void Simulator::initDefaultEnvironment() {
    Solver::Environment env;
    env.save(Env::DefaultEnvPath());

    Solver::Configuration cfg;
    cfg.save(Env::DefaultCfgPath());
}

void Simulator::run(const Task &task) {
    String instanceName(task.instSet + task.instId + ".json");
    String solutionName(task.instSet + task.instId + ".json");

    char argBuf[Cmd::MaxArgNum][Cmd::MaxArgLen];
    char *argv[Cmd::MaxArgNum];
    for (int i = 0; i < Cmd::MaxArgNum; ++i) { argv[i] = argBuf[i]; }
    strcpy(argv[ArgIndex::ExeName], ProgramName().c_str());

    int argc = ArgIndex::ArgStart;

    strcpy(argv[argc++], Cmd::InstancePathOption().c_str());
    strcpy(argv[argc++], (InstanceDir() + instanceName).c_str());

    System::makeSureDirExist(SolutionDir());
    strcpy(argv[argc++], Cmd::SolutionPathOption().c_str());
    strcpy(argv[argc++], (SolutionDir() + solutionName).c_str());

    if (!task.randSeed.empty()) {
        strcpy(argv[argc++], Cmd::RandSeedOption().c_str());
        strcpy(argv[argc++], task.randSeed.c_str());
    }

    if (!task.timeout.empty()) {
        strcpy(argv[argc++], Cmd::TimeoutOption().c_str());
        strcpy(argv[argc++], task.timeout.c_str());
    }

    if (!task.maxIter.empty()) {
        strcpy(argv[argc++], Cmd::MaxIterOption().c_str());
        strcpy(argv[argc++], task.maxIter.c_str());
    }

    if (!task.jobNum.empty()) {
        strcpy(argv[argc++], Cmd::JobNumOption().c_str());
        strcpy(argv[argc++], task.jobNum.c_str());
    }

    if (!task.runId.empty()) {
        strcpy(argv[argc++], Cmd::RunIdOption().c_str());
        strcpy(argv[argc++], task.runId.c_str());
    }

    if (!task.cfgPath.empty()) {
        strcpy(argv[argc++], Cmd::ConfigPathOption().c_str());
        strcpy(argv[argc++], task.cfgPath.c_str());
    }

    if (!task.logPath.empty()) {
        strcpy(argv[argc++], Cmd::LogPathOption().c_str());
        strcpy(argv[argc++], task.logPath.c_str());
    }

    Cmd::run(argc, argv);
}

void Simulator::run(const String &envPath) {
    char argBuf[Cmd::MaxArgNum][Cmd::MaxArgLen];
    char *argv[Cmd::MaxArgNum];
    for (int i = 0; i < Cmd::MaxArgNum; ++i) { argv[i] = argBuf[i]; }
    strcpy(argv[ArgIndex::ExeName], ProgramName().c_str());

    int argc = ArgIndex::ArgStart;

    strcpy(argv[argc++], Cmd::EnvironmentPathOption().c_str());
    strcpy(argv[argc++], envPath.c_str());

    Cmd::run(argc, argv);
}

void Simulator::debug() {
    Task task;
    task.instSet = "";
    task.instId = "dsjc.n500e62624c49";
    //task.randSeed = "1500972793";
    task.randSeed = to_string(Random::generateSeed());
    task.timeout = "1800";
    //task.maxIter = "1000000000";
    task.jobNum = "1";
    task.cfgPath = Env::DefaultCfgPath();
    task.logPath = Env::DefaultLogPath();
    task.runId = "0";

    run(task);
}

void Simulator::benchmark(int repeat) {
    Task task;
    task.instSet = "";
    //task.timeout = "180";
    //task.maxIter = "1000000000";
    task.timeout = "3600";
    //task.maxIter = "1000000000";
    task.jobNum = "1";
    task.cfgPath = Env::DefaultCfgPath();
    task.logPath = Env::DefaultLogPath();

    random_device rd;
    mt19937 rgen(rd());
    for (int i = 0; i < repeat; ++i) {
        //shuffle(instList.begin(), instList.end(), rgen);
        for (auto inst = instList.begin(); inst != instList.end(); ++inst) {
            task.instId = *inst;
            task.randSeed = to_string(Random::generateSeed());
            task.runId = to_string(i);
            run(task);
        }
    }
}

void Simulator::parallelBenchmark(int repeat) {
    Task task;
    task.instSet = "";
    //task.timeout = "180";
    //task.maxIter = "1000000000";
    task.timeout = "3600";
    //task.maxIter = "1000000000";
    task.jobNum = "1";
    task.cfgPath = Env::DefaultCfgPath();
    task.logPath = Env::DefaultLogPath();

    ThreadPool<> tp(4);

    random_device rd;
    mt19937 rgen(rd());
    for (int i = 0; i < repeat; ++i) {
        //shuffle(instList.begin(), instList.end(), rgen);
        for (auto inst = instList.begin(); inst != instList.end(); ++inst) {
            task.instId = *inst;
            task.randSeed = to_string(Random::generateSeed());
            task.runId = to_string(i);
            tp.push([=]() { run(task); });
            this_thread::sleep_for(1s);
        }
    }
}

void Simulator::generateInstance(const InstanceTrait &trait) {
    Random rand;

    Problem::Input input;

    input.set_rownuma(trait.dim);
    input.set_numrcab(trait.dim);
    input.set_colnumb(trait.dim);
    input.set_refmultiplicationnum(trait.refMultiplicationNum);

    ostringstream path;
    path << InstanceDir() << "square." << input.rownuma() << "x" << input.numrcab() << "x" << input.colnumb()
        << "m" << input.refmultiplicationnum() << ".json";
    save(path.str(), input);
}

void Simulator::generateInstances() {
    InstanceTrait trait;

    vector<Interval<int>> ref = {
        {0, -1}, // 0
        {0, -1}, // 1
        {6, 7},  // 2
        {21, 23} // 3
    };

    for (int i = 0; i < ref.size(); ++i) {
        trait.dim = i;
        for (int m = ref[i].end; m >= ref[i].begin; --m) {
            trait.refMultiplicationNum = m;
            generateInstance(trait);
        }
    }

    //for (int i = 4; i <= 16; ++i) {
    //    trait.dim = i;
    //    trait.refMultiplicationNum = i * i * i;
    //    generateInstance(trait);
    //}
}

}
