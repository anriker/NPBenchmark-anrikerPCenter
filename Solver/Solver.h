////////////////////////////////
/// usage : 1.	the entry to the algorithm for the guillotine cut problem.
/// 
/// note  : 1.	
////////////////////////////////

#ifndef SMART_SZX_P_CENTER_SOLVER_H
#define SMART_SZX_P_CENTER_SOLVER_H


#include "Config.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <thread>

#include "Common.h"
#include "Utility.h"
#include "LogSwitch.h"
#include "Problem.h"
#define MAX_ITER 1000
#define count 2
#define RECENCY 5
//#define numPcenter 10
#define MAX 10000000
#define MIN -1000


namespace szx {
const int INF = 1000;
class Solver {
    int best_solution;//Ŀ�꺯��ֵ����������
    int flag;
    int badcount;
    std::vector <std::vector<int>> TabuTenure;
    std::vector<int> pcenter;
    std::vector<int> sameScid;
    std::vector <std::vector <int>> PtoNode;
    std::vector <std::vector <int>> F;
    std::vector <std::vector <int>> D;
    int iter;
    struct pair {
        int nodeid;
        int centerid;
        int delt;
    };
    struct Scinfo {
        int Scid;
        int Sc;
    };
    #pragma region Type
public:
    // commmand line interface.
    struct Cli {
        static constexpr int MaxArgLen = 256;
        static constexpr int MaxArgNum = 32;

        static String InstancePathOption() { return "-p"; }
        static String SolutionPathOption() { return "-o"; }
        static String RandSeedOption() { return "-s"; }
        static String TimeoutOption() { return "-t"; }
        static String MaxIterOption() { return "-i"; }
        static String JobNumOption() { return "-j"; }
        static String RunIdOption() { return "-rid"; }
        static String EnvironmentPathOption() { return "-env"; }
        static String ConfigPathOption() { return "-cfg"; }
        static String LogPathOption() { return "-log"; }

        static String AuthorNameSwitch() { return "-name"; }
        static String HelpSwitch() { return "-h"; }

        static String AuthorName() { return "szx"; }
        static String HelpInfo() {
            return "Pattern (args can be in any order):\n"
                "  exe (-p path) (-o path) [-s int] [-t seconds] [-name]\n"
                "      [-iter int] [-j int] [-id string] [-h]\n"
                "      [-env path] [-cfg path] [-log path]\n"
                "Switches:\n"
                "  -name  return the identifier of the authors.\n"
                "  -h     print help information.\n"
                "Options:\n"
                "  -p     input instance file path.\n"
                "  -o     output solution file path.\n"
                "  -s     rand seed for the solver.\n"
                "  -t     max running time of the solver.\n"
                "  -i     max iteration of the solver.\n"
                "  -j     max number of working solvers at the same time.\n"
                "  -rid   distinguish different runs in log file and output.\n"
                "  -env   environment file path.\n"
                "  -cfg   configuration file path.\n"
                "  -log   activate logging and specify log file path.\n"
                "Note:\n"
                "  0. in pattern, () is non-optional group, [] is optional group\n"
                "     when -env option is not given.\n"
                "  1. an environment file contains information of all options.\n"
                "     explicit options get higher priority than this.\n"
                "  2. reaching either timeout or iter will stop the solver.\n"
                "     if you specify neither of them, the solver will be running\n"
                "     for a long time. so you should set at least one of them.\n"
                "  3. the solver will still try to generate an initial solution\n"
                "     even if the timeout or max iteration is 0. but the solution\n"
                "     is not guaranteed to be feasible.\n";
        }

        // a dummy main function.
        static int run(int argc, char *argv[]);
    };

    // controls the I/O data format, exported contents and general usage of the solver.
    struct Configuration {
        enum Algorithm { Greedy, TreeSearch, DynamicProgramming, LocalSearch, Genetic, MathematicallProgramming };


        Configuration() {}

        void load(const String &filePath);
        void save(const String &filePath) const;


        String toBriefStr() const {
            String threadNum(std::to_string(threadNumPerWorker));
            std::ostringstream oss;
            oss << "alg=" << alg
                << ";job=" << threadNum;
            return oss.str();
        }


        Algorithm alg = Configuration::Algorithm::Greedy; // OPTIMIZE[szx][3]: make it a list to specify a series of algorithms to be used by each threads in sequence.
        int threadNumPerWorker = (std::min)(1, static_cast<int>(std::thread::hardware_concurrency()));
    };

    // describe the requirements to the input and output data interface.
    struct Environment {
        static constexpr int DefaultTimeout = (1 << 30);
        static constexpr int DefaultMaxIter = (1 << 30);
        static constexpr int DefaultJobNum = 0;
        // preserved time for IO in the total given time.
        static constexpr int SaveSolutionTimeInMillisecond = 1000;

        static constexpr Duration RapidModeTimeoutThreshold = 600 * static_cast<Duration>(Timer::MillisecondsPerSecond);

        static String DefaultInstanceDir() { return "Instance/"; }
        static String DefaultSolutionDir() { return "Solution/"; }
        static String DefaultVisualizationDir() { return "Visualization/"; }
        static String DefaultEnvPath() { return "env.csv"; }
        static String DefaultCfgPath() { return "cfg.csv"; }
        static String DefaultLogPath() { return "log.csv"; }

        Environment(const String &instancePath, const String &solutionPath,
            int randomSeed = Random::generateSeed(), double timeoutInSecond = DefaultTimeout,
            Iteration maxIteration = DefaultMaxIter, int jobNumber = DefaultJobNum, String runId = "",
            const String &cfgFilePath = DefaultCfgPath(), const String &logFilePath = DefaultLogPath())
            : instPath(instancePath), slnPath(solutionPath), randSeed(randomSeed),
            msTimeout(static_cast<Duration>(timeoutInSecond * Timer::MillisecondsPerSecond)), maxIter(maxIteration),
            jobNum(jobNumber), rid(runId), cfgPath(cfgFilePath), logPath(logFilePath), localTime(Timer::getTightLocalTime()) {}
        Environment() : Environment("", "") {}

        void load(const Map<String, char*> &optionMap);
        void load(const String &filePath);
        void loadWithoutCalibrate(const String &filePath);
        void save(const String &filePath) const;

        void calibrate(); // adjust job number and timeout to fit the platform.

        String solutionPathWithTime() const { return slnPath + "." + localTime; }

        String visualizPath() const { return DefaultVisualizationDir() + friendlyInstName() + "." + localTime + ".html"; }
        template<typename T>
        String visualizPath(const T &msg) const { return DefaultVisualizationDir() + friendlyInstName() + "." + localTime + "." + std::to_string(msg) + ".html"; }
        String friendlyInstName() const { // friendly to file system (without special char).
            auto pos = instPath.find_last_of('/');
            return (pos == String::npos) ? instPath : instPath.substr(pos + 1);
        }
        String friendlyLocalTime() const { // friendly to human.
            return localTime.substr(0, 4) + "-" + localTime.substr(4, 2) + "-" + localTime.substr(6, 2)
                + "_" + localTime.substr(8, 2) + ":" + localTime.substr(10, 2) + ":" + localTime.substr(12, 2);
        }

        // essential information.
        String instPath;
        String slnPath;
        int randSeed;
        Duration msTimeout;

        // optional information. highly recommended to set in benchmark.
        Iteration maxIter;
        int jobNum; // number of solvers working at the same time.
        String rid; // the id of each run.
        String cfgPath;
        String logPath;

        // auto-generated data.
        String localTime;
    };

    struct Solution : public Problem::Output { // cutting patterns.
        Solution(Solver *pSolver = nullptr) : solver(pSolver) {}

        Solver *solver;
    };
    #pragma endregion Type

    #pragma region Constant
public:
    #pragma endregion Constant

    #pragma region Constructor
public:
    Solver(const Problem::Input &inputData, const Environment &environment, const Configuration &config)
        : input(inputData), env(environment), cfg(config), rand(environment.randSeed),
        timer(std::chrono::milliseconds(environment.msTimeout)), iteration(1) {}
    #pragma endregion Constructor

    #pragma region Method
public:
    bool solve(); // return true if exit normally. solve by multiple workers together.
    bool check(Length &obj) const;
    void record() const; // save running log.

protected:
    void init();
    bool optimize(Solution &sln, unsigned int &object, ID workerId = 0); // optimize by a single worker.
    int init_findP(Solution &sln, Scinfo &ScInfo, ID nodeNum);//�ҵ���Զ����߶�Ӧ�������С����Զ����ߵĵ㣬���Ѱ����һ��P��
    bool init_solution(Solution &sln, Scinfo &ScInfo, ID nodeNum, ID centerNum);//��ʼ��
    bool initfuncation(Solution &sln, Scinfo &ScInfo,ID nodeNum);//��ʼĿ�꺯��ֵ
    void initDandFtable(Solution &sln, int tempP, ID nodeNum);
    int find_sameScid(Solution &sln, Scinfo &ScInfo,ID nodeNum);
    int find_pair(Solution &sln, Scinfo &ScInfo, pair &Pair, ID nodeNum,ID centerNum);
    void add_facility(Solution &sln, pair &Pair, ID nodeNum);//���������f������F���D��
    void remove_facility(Solution &sln, pair &Pair, ID nodeNum,ID centerNum);//ɾ�������f������F���D��
    //int fine_move(Solution &sln);//Ѱ�������ڵ�i������ʵ�֣�
    void change_pair(Solution &sln, pair &Pair, ID nodeNum, ID centerNum);//�Ƚ�Ŀ�꺯��ֵSc�������ڵ������ľ���Mf��ȡ���ߡ�
    int find_next(Solution &sln, int v, ID nodeNum,ID centerNum);
    #pragma endregion Method

    #pragma region Field
public:
    Problem::Input input;
    Problem::Output output;

    struct { // auxiliary data for solver.
        double objScale;

        Arr2D<Length> adjMat; // adjMat[i][j] is the distance of the edge which goes from i to j.
        Arr<Length> coverRadii; // coverRadii[n] is the length of its shortest serve arc.
    } aux;

    Environment env;
    Configuration cfg;

    Random rand; // all random number in Solver must be generated by this.
    Timer timer; // the solve() should return before it is timeout.
    Iteration iteration;
    #pragma endregion Field
}; // Solver 

}


#endif // SMART_SZX_P_CENTER_SOLVER_H
