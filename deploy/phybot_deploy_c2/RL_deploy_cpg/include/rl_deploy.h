

#include "DataPackage/include/DataPackage.h"
#include <torch/script.h>
#include <fstream>
#include <iomanip>

class rl_deploy_cpg{

public:

    rl_deploy_cpg();
    ~rl_deploy_cpg();

    // void Init();

    void GetDataFromPackage(DataPackage &data);
    void SetDataToPackage(DataPackage &data);
    void Step();
    void Exit();
    Eigen::MatrixXd LoadMotionLib(const std::string& filename, bool skipHeader);
    Eigen::Vector3d Euler_ZYXToGravityVec(Eigen::Vector3d euler_zyx);

    // Additive, non-behavioral accessor exposing the 283-D observation vector
    // (inputdata_eigen) for bit-exact golden-fixture verification. Read-only.
    const Eigen::VectorXd& observation() const { return inputdata_eigen; }



private:
    torch::jit::script::Module policy;
    Eigen::Vector3d gravity_vec_eigen;
    Eigen::VectorXd default_dof_pos_eigen;
    Eigen::VectorXd last_actions_eigen;
    Eigen::VectorXd inputdata_eigen;
    Eigen::VectorXd outputdata_eigen;

    Eigen::VectorXd pos_hist_buf_eigen;
    Eigen::VectorXd vel_hist_buf_eigen;
    Eigen::VectorXd action_hist_buf_eigen;
    Eigen::VectorXd rot_hist_buf_eigen;
    Eigen::VectorXd ang_vel_hist_buf_eigen;

    Eigen::VectorXd q_origin;
    Eigen::VectorXd dot_q_origin;

    Eigen::VectorXd tor_origin;
    Eigen::VectorXd sim_tor;
    Eigen::VectorXd sim_tor_p;
    Eigen::VectorXd sim_tor_d;

    Eigen::VectorXd lin_vel;           
    Eigen::VectorXd imu_angular_vel;      
    Eigen::VectorXd commands;
    // Eigen::VectorXd task_commands;    
    Eigen::VectorXd base_euler;   
    Eigen::VectorXd base_quat;   
    Eigen::VectorXd dof_pos;           
    Eigen::VectorXd dof_vel;           
    Eigen::VectorXd last_actions;
    Eigen::VectorXd task_obs;

    torch::Tensor torque_limits;
    torch::Tensor commands_scale;
    torch::Tensor default_dof_pos;
    torch::Tensor output_dof_pos;

    int decimation;
    int num_proprio;
    int num_observations;

    int num_js_commands;
    int num_mimic_commands;
    int num_commands;
    double clip_obs;
    double clip_actions_lower;
    double clip_actions_upper;
    // double action_scale;
    int num_of_dofs;
    int history_len;
    double lin_vel_scale;
    double ang_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    double js_vx_desire;
    double js_vy_desire;
    double js_OmegaZ_desire;
    int step_num = 0;
    int episode_length = 0;
    int frame_count = 0;
    int num_motion_frames;



    double phase;
    std::string policy_path;
    std::string mimic_policy_path;
    std::string motion_lib_path;
    std::vector<int> indices_to_remove;

    Eigen::VectorXd rl_p;
    Eigen::VectorXd rl_d;
    Eigen::VectorXd action_scale;

    template<typename T>
    std::vector<T> ReadVectorFromYaml(const YAML::Node& node)
    {
        std::vector<T> values;
        for(const auto& val : node)
        {
            values.push_back(val.as<T>());
        }
        return values;
    }

    Eigen::MatrixXd motion_frames;
    Eigen::VectorXd remove_indices(const Eigen::VectorXd& input, const std::vector<int>& indices_to_remove);
    void assign_with_skipped_zero(Eigen::VectorXd& target,const Eigen::VectorXd& source,const std::vector<int>& indices_to_skip);

    std::ofstream task_obs_log_file;
    bool enable_task_obs_logging;

    void InitTaskObsLogging(const std::string& log_file_path = "../RL_deploy/logs/task_obs_log.csv");
    void LogTaskObs(const torch::Tensor& task_obs_tensor);
    void CloseTaskObsLogging();

    // Adding

    #define COUPLING_CHANGE_RATE_IN 4.0
    #define COUPLING_CHANGE_RATE_DE 1.0
    #define AMPLITUDE_CHANGE_RATE_IN 1.05
    #define AMPLITUDE_CHANGE_RATE_DE 0.995

    int num_feet = 4;
    double step_dt;
    double gait_cy;
    Eigen::Vector4d cycle_r;
    double cycle_r_max = 1.0;
    double cycle_r_min = 0.001;
    Eigen::Vector4d px;
    Eigen::Vector4d py;
    double rate = 10.0;
    Eigen::Vector4d phase_offset;
    Eigen::Vector4d phase_offset_target;
    double phase_offset_change_rate;
    Eigen::Vector4d phase_offset_stand;
    Eigen::Vector4d phase_offset_walk;
    Eigen::Vector4d phase_offset_run;
    Eigen::Vector4d coupling_change_rate;

    Eigen::Vector4d contact_ratio;
    double coff_b = 1000.0;
    double coupling_change_rate_increase;
    double coupling_change_rate_decrease;
    Eigen::Vector4d amplitude_change_rate;
    double amplitude_change_rate_increase;
    double amplitude_change_rate_decrease;

    bool walk_flag = false;
    bool stand_flag = false;

    
    void CPGInit(int num_feet, float dt, float frequency);
    Eigen::VectorXd CPGGammaCal();
    Eigen::MatrixXd CPGCoupling();
    void CPGPhaseChange();
    void CPGHopfOscillator();
    void CPGReset(bool flag);
    void CPGSetCy(Eigen::VectorXd Cycle_r);
    void CPGSetCou(Eigen::VectorXd Coupling_change_rate);
    void CPGSetA(Eigen::VectorXd Amplitude_change_rate);
    void CPGSetP(Eigen::VectorXd Phase_offset);
    void CPGSetC(Eigen::VectorXd Contact_ratio);
    void CPGSetncy(double frequency);
    void CPGSetSn();

    void CPGSetWP_1();
    void CPGSetWP_2();
    void CPGSetWP_3();
    void CPGSetWP_4();
    void CPGSetWP_5();
    void CPGSetWP_6();
    void CPGSetWP_7();
    void CPGSetWP_8();

    double flag_1 = 0.05;
    double flag_2 = 0.3;
    double flag_3 = 0.6;
    double flag_4 = 0.9;
    double flag_5 = 1.2;
    double flag_6 = 1.5;
    double flag_7 = 1.8;
    double flag_8 = 2.1;
    double flag_9 = 3.0;

    

    Eigen::VectorXd CPGGetXNorm();
    Eigen::VectorXd CPGGetYNorm();
    void UpdateGaitGeneratorPattern(Eigen::VectorXd Commands);
    void CommandRefinement();


    Eigen::VectorXd dataL;
    std::ofstream foutData;
    bool dataLog(Eigen::VectorXd &v, std::ofstream &f);
};


