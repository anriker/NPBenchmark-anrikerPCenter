#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>


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

    fstream file;
    float object = 127;
    string name;
    file.open("best_result.txt", ios::in | ios::out);
    if (!file) {
        cout << "**error" << endl;
    }
    char *line = new char[INF];
    float obj;
    while (!file.eof()) {
        file >> obj;
        file >> name;
        if ("Instance/" + name + ".json" == env.instPath) {
            object = obj;
           // cout<<object<<endl;
            break;
        }
    }file.close();

    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], object, i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Length bestValue = Problem::MaxDistance;
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].coverRadius << endl;
        if (solutions[i].coverRadius >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].coverRadius;
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

    Length obj = output.coverRadius;
    Length checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ",";
        if (Problem::isTopologicalGraph(input)) {
            log << obj << ",";
        } else {
            auto oldPrecision = log.precision();
            log.precision(2);
            log << fixed << setprecision(2) << (obj / aux.objScale) << ",";
            log.precision(oldPrecision);
        }
        log << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    // EXTEND[szx][2]: save solution in log.
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Distance,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
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
        TooManyCentersError = 0x2
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::TooManyCentersError) { Log(LogSwitch::Checker) << "TooManyCentersError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {
    ID nodeNum = input.graph().nodenum();

    aux.adjMat.init(nodeNum, nodeNum);
    fill(aux.adjMat.begin(), aux.adjMat.end(), Problem::MaxDistance);
    for (ID n = 0; n < nodeNum; ++n) { aux.adjMat.at(n, n) = 0; }

    if (Problem::isTopologicalGraph(input)) {
        aux.objScale = Problem::TopologicalGraphObjScale;
        for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
            // only record the last appearance of each edge.
            aux.adjMat.at(e->source(), e->target()) = e->length();
            aux.adjMat.at(e->target(), e->source()) = e->length();
        }

        Timer timer(30s);
        constexpr bool IsUndirectedGraph = true;
        IsUndirectedGraph
            ? Floyd::findAllPairsPaths_symmetric(aux.adjMat)
            : Floyd::findAllPairsPaths_asymmetric(aux.adjMat);
        Log(LogSwitch::Preprocess) << "Floyd takes " << timer.elapsedSeconds() << " seconds." << endl;
    } else { // geometrical graph.
        aux.objScale = Problem::GeometricalGraphObjScale;
        for (ID n = 0; n < nodeNum; ++n) {
            double nx = input.graph().nodes(n).x();
            double ny = input.graph().nodes(n).y();
            for (ID m = 0; m < nodeNum; ++m) {
                if (n == m) { continue; }
                aux.adjMat.at(n, m) = static_cast<Length>(aux.objScale * hypot(
                    nx - input.graph().nodes(m).x(), ny - input.graph().nodes(m).y()));//将距离增大两倍，化成整数
            }
        }
    }

    aux.coverRadii.init(nodeNum);
    fill(aux.coverRadii.begin(), aux.coverRadii.end(), Problem::MaxDistance);
}

bool Solver::init_solution(Solution &sln, Scinfo &ScInfo, ID nodeNum, ID centerNum) {
    srand(rand() % MAX);
    pcenter.push_back(rand() % nodeNum);
    for (int i = 0; i != nodeNum; i++) {
        D[i][0] = aux.adjMat[pcenter[0]][i];
        F[i][0] = pcenter[0];
    }
    int i = initfuncation(sln, ScInfo, nodeNum);
    if (i == -1)
        return -1;
    for (int i = 1; i != centerNum; i++) {
        int tempP = init_findP(sln, ScInfo, nodeNum);
        initDandFtable(sln, tempP, nodeNum);
        pcenter.push_back(tempP);
        initfuncation(sln, ScInfo, nodeNum);
    }
    return 1;
}

int Solver::init_findP(Solution &sln, Scinfo &ScInfo, ID nodeNum) {
    vector <int> id;
    int tempP;
    for (int i = 0; i != nodeNum; i++) {
        if (aux.adjMat[ScInfo.Scid][i] < ScInfo.Sc) {
            id.push_back(i);
        }
    }
    int tempi = 0;
    if (id.size() > 1) {
        srand(time(0));
        tempi = rand() % id.size();
    }
    tempP = id[tempi];
    return tempP;
}

void Solver::initDandFtable(Solution &sln, int tempP, ID nodeNum) {

    for (int j = 0; j != nodeNum; j++) {
        if (pcenter.size() == 1) {
            if (aux.adjMat[pcenter[0]][j] < aux.adjMat[tempP][j]) {
                D[j][0] = aux.adjMat[pcenter[0]][j];
                F[j][0] = pcenter[0];
                D[j][1] = aux.adjMat[tempP][j];
                F[j][1] = tempP;
            } else {
                D[j][0] = aux.adjMat[tempP][j];
                F[j][0] = tempP;
                D[j][1] = aux.adjMat[pcenter[0]][j];
                F[j][1] = pcenter[0];
            }
        } else {
            if (aux.adjMat[tempP][j] < D[j][0]) {
                D[j][1] = D[j][0];
                F[j][1] = F[j][0];
                D[j][0] = aux.adjMat[tempP][j];
                F[j][0] = tempP;

            } else if (aux.adjMat[tempP][j] < D[j][1]) {
                D[j][1] = aux.adjMat[tempP][j];
                F[j][1] = tempP;
            }

        }
    }
}


bool Solver::optimize(Solution &sln, float &object, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;
    bool status = true;
    ID nodeNum = input.graph().nodenum();
    ID centerNum = input.centernum();
    //sln.maxLength = 0;
    //auto &centers(*sln.mutable_centers());
   // centers.Resize(centerNum, Problem::InvalidId);
    TabuTenure = vector<vector<int>>(nodeNum, vector<int>(nodeNum, 0));
    F = vector<vector<int>>(nodeNum, vector<int>(2, 0));
    D = vector<vector<float>>(nodeNum, vector<float>(2, 0));

    // TODO[0]: replace the following random assignment with your own algorithm.
    //for (int e = 0; !timer.isTimeOut() && (e < centerNum); ++e) { // 随机生成服务节点p
    //    int index = rand.pick(0, nodeNum);
    //    sln.add_centers(index);
    //}

    //vector<int> serverLengthList; // 存储所有服务边
    //serverLengthList.reserve(nodeNum);
    //for (int i = 0; i < nodeNum; ++i) {
    //    int serveLength = INF; // 节点的服务边长度
    //    for (int j = 0; j < centerNum; ++j) {
    //        int k = sln.centers(j);
    //        if (serveLength > aux.adjMat[i][k]) {
    //            serveLength = aux.adjMat[i][k];
    //        }
    //    }
    //    serverLengthList.push_back(serveLength);
    //}
    //auto maxPosition_s = max_element(serverLengthList.begin(), serverLengthList.end());
    //sln.maxLength = *maxPosition_s; // 所有服务边中的最大值，即问题的输出
    pair Pair = { -1 };
    Scinfo ScInfo = { -1 };
    int flag = init_solution(sln, ScInfo, nodeNum, centerNum);//初始解
    if (flag == -1) {
        cout << "error!!";
        return -1;
    }

    initfuncation(sln, ScInfo, nodeNum);
    best_solution = ScInfo.Sc;
    /*for (int i = 0; i != pcenter.size(); i++) {
        cout << pcenter[i] + 1 << " ";
    }*/
    cout << endl << "the init best solution:" << best_solution << endl;
    clock_t start_time = clock();
    iter = 1;
    clock_t mid_tim;
    while ((mid_tim - start_time)*1.0 / CLOCKS_PER_SEC < 200)//搜索条件
    {
        flag = 1;
        Pair = { -1 };
        int flag = find_pair(sln, ScInfo, Pair, nodeNum, centerNum);//tabu发现交换对
       // cout << "Pair:" << Pair.nodeid << "," << Pair.centerid << "," << Pair.delt << endl;
        //if (Pair.centerid == -1 || Pair.nodeid == -1)
           // break;      
        change_pair(sln, Pair, nodeNum, centerNum);//更新交换对 
        initfuncation(sln, ScInfo, nodeNum);
        if (best_solution > ScInfo.Sc) {
            best_solution = ScInfo.Sc;
           // cout << best_solution <<"\t";
        }
        iter++;
        if (ScInfo.Sc == object)
            break;
        //cout  << ScInfo.Sc << "\t";
        mid_tim = clock();
    }
   /* for (int i = 0; i != nodeNum; i++) {
        for (int j = 0; j != 2; j++) {
            cout << "F[" << i << "]" << "[" << j << "]=" << F[i][j] << "D[" << i << "]" << "[" << j << "]=" << D[i][j] << "\t";
        }cout << endl;
    }cout << endl;*/
    clock_t end_time = clock();
    for (int i = 0; i != pcenter.size(); i++) {
        cout << pcenter[i] << " ";
        sln.add_centers(pcenter[i]);
    }
    if (ScInfo.Sc < best_solution)
        best_solution = ScInfo.Sc;
    //sln.maxLength = best_solution;
    cout << "the iter: " << iter << endl;
    cout << "the most solution: " << best_solution << endl;
    cout << "the true best solution:" << object << endl;
    cout << "the time is:" << (end_time - start_time)*1.0 / CLOCKS_PER_SEC << "s" << endl;
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}

bool Solver::initfuncation(Solution &sln, Scinfo &ScInfo, ID nodeNum) {
    int tempid;
    float tempSc = -1;
    for (int j = 0; j != nodeNum; j++) {//最长边对应的服务点有多个时应该随机选择一个
        if (tempSc < D[j][0]) {
            tempSc = D[j][0];
            tempid = j;
        } else if (tempSc == D[j][0]) {
            if (rand() % 2 == 0) {
                tempid = j;
            }
        }
    }
    //cout << tempSc << endl;
    ScInfo.Sc = tempSc;
    ScInfo.Scid = tempid;
    return 1;//是否应该有一个数组记录当前所有最长服务边对应的普通节点
}

int Solver::find_pair(Solution &sln, Scinfo &ScInfo, pair &Pair, ID nodeNum, ID centerNum) {//确定交换对<nodei，centerj>
    vector <int> id;//记录小于最长边的点对应的小于最长服务边的点
    vector <pair> tempPair;
    pair tabu_pair;// = { 0 };
    tabu_pair.delt = MAX;
    pair no_tabu_pair;// = { 0 };
    no_tabu_pair.delt = MAX;
    int no_tabu_samenumber = 0;
    int tabu_samenumber = 0;
    int same_tabuNum = 0, same_no_tabuNum = 0;
    int tempcount = 0;
    int m = 0;
    for (int i = 0; i != nodeNum; i++) {
        if (aux.adjMat[ScInfo.Scid][i] < ScInfo.Sc) {
            id.push_back(i);
        }
    }
    if (id.size() == 0)
        return -1;
    vector <Scinfo> tempScinfo((id.size()));
    for (int i = 0; i != id.size(); i++) {
        //找最好的交换对
        pair tempPair1;
        tempPair1.delt = MAX;
        tempPair1.nodeid = id[i];
        vector <vector <int> > tempF(F);
        vector <vector <float> > tempD(D);
        float tempSc = -1;
        for (int j = 0; j != nodeNum; j++) {//添加了服务点id[i]
            if (tempD[j][0] > aux.adjMat[id[i]][j]) {
                tempD[j][1] = tempD[j][0];
                tempF[j][1] = tempF[j][0];
                tempD[j][0] = aux.adjMat[id[i]][j];
                tempF[j][0] = id[i];
            } else if (tempD[j][1] > aux.adjMat[id[i]][j]) {
                tempD[j][1] = aux.adjMat[id[i]][j];
                tempF[j][1] = id[i];
            }
            if (tempD[j][0] > tempSc)
                tempSc = tempD[j][0];
        }
        //cout<<tempSc << "/t";
        //vector <float> Mf(centerNum, 0);
        float Mf;
        for (int t = 0; t != centerNum; t++) {//寻找加入节点id[i]后删除中心节点pcenter[t]得到的最长服务边
            Mf = -1;
            for (int j = 0; j != nodeNum; j++) {
                if (tempF[j][0] == pcenter[t])
                    if (Mf < tempD[j][1]) {//&& Mf[t] < tempD[j][1]) {
                        Mf = tempD[j][1];
                    }
            }//cout << Mf << "**";
            if (TabuTenure[pcenter[t]][id[i]] <= iter) {//update the no_tabu best move,有错误;
                if (max(Mf, tempSc) < no_tabu_pair.delt) {
                    no_tabu_pair = { id[i],pcenter[t],max(Mf,tempSc) };
                    same_no_tabuNum = 1;
                } else if (max(Mf, tempSc) == no_tabu_pair.delt) {
                    same_no_tabuNum++;
                    if (rand() % same_no_tabuNum == 0)
                        no_tabu_pair = { id[i],pcenter[t],max(Mf,tempSc) };
                }
            } else {//update the tabu best move;
                if (max(Mf, tempSc) < tabu_pair.delt) {
                    tabu_pair = { id[i],pcenter[t],max(Mf,tempSc) };
                    same_tabuNum = 1;
                } else if (max(Mf, tempSc) == tabu_pair.delt) {
                    same_tabuNum++;
                    if (rand() % same_tabuNum == 0)
                        tabu_pair = { id[i],pcenter[t],max(Mf,tempSc) };
                }
            }

        }//cout << endl;
        tempF.clear();
        tempD.clear();

    }
   // cout << "haha" << endl;
    if ((tabu_pair.delt < no_tabu_pair.delt) && tabu_pair.delt < best_solution)// && (tabu_pair.delt > 0.01)) //解禁条件：禁忌解优于当前查找的非禁忌解中最好的且优于历史最优解
    {
        if (Pair.centerid == tabu_pair.nodeid && Pair.nodeid == tabu_pair.centerid) {
            return -1;
        }
        Pair = tabu_pair;
        cout << "** ";
    } else {//if (no_tabu_pair.delt <= ScInfo.Sc) {

        Pair = no_tabu_pair;
    } /*if (Pair.centerid == -1 || Pair.nodeid == -1) {
        ScInfo.Scid = find_sameScid(sln, ScInfo, nodeNum);
        find_pair(sln, ScInfo, Pair, nodeNum, centerNum);
    }*/
   // cout << "tempSC:" << Pair.delt << endl;
    return 1;

}

int Solver::find_sameScid(Solution &sln, Scinfo &ScInfo, ID nodeNum) {
    for (int i = 0; i != nodeNum; i++) {
        if (D[i][0] == ScInfo.Sc) {
            sameScid.push_back(i);
        }
    }
    int t = rand() % sameScid.size();
    while (sameScid[t] == ScInfo.Scid)
        t = rand() % sameScid.size();
    return sameScid[t];
}

void Solver::add_facility(Solution &sln, pair &Pair, ID nodeNum) {
    for (int i = 0; i != nodeNum; i++) {
        if (D[i][0] > aux.adjMat[Pair.nodeid][i]) {
            D[i][1] = D[i][0];
            F[i][1] = F[i][0];
            D[i][0] = aux.adjMat[Pair.nodeid][i];
            F[i][0] = Pair.nodeid;

        } else if (pcenter.size() > 1 && D[i][1] > aux.adjMat[Pair.nodeid][i]) {
            D[i][1] = aux.adjMat[Pair.nodeid][i];
            F[i][1] = Pair.nodeid;
        }

    }
}

#if 0
int Tabu::findremove_facility(vector <Nodes> &Node, int f) {//前p个服务点中找到一个删除后产生的最大服务边最小的点删(禁忌策略找交换对<f,i>）
    vector <int> Mf(centerNum, 0);
    float tempSc = 0;
    int tempScid;
    for (int j = 0; j != centerNum; j++) {
        for (int i = 0; i != nodeNum; i++) {
            if (F[Node[i].id][0] == pcenter[j] && Mf[j] < D[Node[i].id][1]) {
                Mf[j] = D[Node[i].id][1];
            }
        }
    }
    tempSc = Mf[0];
    for (int j = 1; j != centerNum; j++) {
        if (tempSc > Mf[j]) {
            tempSc = Mf[j];
            tempScid = j;
        } else if (tempSc == Mf[j]) {
            if (rand() % 2 == 0) {//相等时，随机选择
                tempSc = Mf[j];
                tempScid = j;
            }
        }
    }
    int i = pcenter[tempScid];
    //更新Sc
    if (tempSc > Sc) Sc = tempSc;
    return i;
    //return tempScid;//找到删除边在数组中的位置
}
#endif

void Solver::remove_facility(Solution &sln, pair &Pair, ID nodeNum, ID centerNum) {
    int tempPlace = 0;
    for (int j = 0; j != centerNum; j++) {
        if (pcenter[j] == Pair.centerid) {
            //tempPlace = j;
            pcenter[j] = Pair.nodeid;
            break;
        }
    }
    for (int i = 0; i != nodeNum; i++) {
        if (F[i][0] == Pair.centerid) {
            D[i][0] = D[i][1];
            F[i][0] = F[i][1];
            int nextp = find_next(sln, i, nodeNum, centerNum);
           // cout  << "nextp1:" << nextp;
            D[i][1] = aux.adjMat[nextp][i];
            F[i][1] = nextp;

        } else if (pcenter.size() > 1 && F[i][1] == Pair.centerid) {
            int nextp = find_next(sln, i, nodeNum, centerNum);
            D[i][1] = aux.adjMat[nextp][i];
            F[i][1] = nextp;
        }
    }
    /*cout << "F,D:" << endl;

    for (int j = 0; j != nodeNum; j++) {
        for (int i = 0; i != 2; i++) {
            cout << F[j][i] << "," << D[j][i] << ";";
        }
        cout << endl;
    }*/

    //pcenter[centerNum + 1] = -1;
}
/*Mf为删除节点f后产生的最长服务边距离（不是当前历史最长距离），
对比所有服务边删除后产生的Mf，选择最小的删除其对应的点,
利用F和D表计算
*/


int Solver::find_next(Solution &sln, int v, ID nodeNum, ID centerNum) {
    float tempsecondmax = MAX;
    int tempsecondmaxP = -1;
    for (int j = 0; j != centerNum; j++) {
         //cout << pcenter[j] << "\t";

        if (pcenter[j] != F[v][0]) {
            if (tempsecondmax > aux.adjMat[pcenter[j]][v]) {
                tempsecondmax = aux.adjMat[pcenter[j]][v];
                tempsecondmaxP = pcenter[j];
            }// else if (tempsecondmax == aux.adjMat[pcenter[j]][v]) {
            //    if (rand() % 2 == 0) {
            //        tempsecondmaxP = pcenter[j];
            //    }
            //}

        }

    }//cout << endl;
    return tempsecondmaxP;
}


void Solver::change_pair(Solution &sln, pair &Pair, ID nodeNum, ID centerNum) {
    //更新tabu表
    //cout << "centerid,nodeid:" << Pair.centerid << "," << Pair.nodeid << endl;
    
    TabuTenure[Pair.centerid][Pair.nodeid] = TabuTenure[Pair.nodeid][Pair.centerid] = 50 + iter + rand() % iter;

    add_facility(sln, Pair, nodeNum);//添加Pair.nodeid服务点
    remove_facility(sln, Pair, nodeNum, centerNum);//删除Pair.centerid服务点
}

//void Solver::check(Solution &sln) {
//   // cout << best_solution << endl;
//    for (int i = 0; i != nodeNum; i++) {
//        if (D[i][0] > best_solution) {
//            cout << "node " << i - 1 << " error," << "the dis is:" << D[i][0] << endl;
//        }
//    }
//}
#pragma endregion Solver

}