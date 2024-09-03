%% 0. Setup
% Ensure MARRMoT is available
marrmot_path = "/Users/wmk934/code/marrmot/MARRMoT";
add_this = genpath(marrmot_path);
addpath(add_this);
clear marrmot_path add_this

% Ensure we're in a known directory so that the relative imports work
cd '/Users/wmk934/data/benchmarking'

% Set the forcing file
forcing_file = "timeseries/hysets_08NB014.csv";

% Set the output folder
output_folder = "./simulations/";
mkdir(output_folder)

%% 1. Prepare data
% Load the forcing data
data = readtable(forcing_file);
dates = data.date;

% Create a climatology data input structure. 
% NOTE: the names of all structure fields are hard-coded in each model
% file. These should not be changed.
input_climatology.precip   = data.total_precipitation_sum;      % Daily data: P rate  [mm/d]
input_climatology.temp     = data.temperature_2m_mean;          % Daily data: mean T  [degree C]
input_climatology.pet      = data.potential_evaporation_sum;    % Daily data: Ep rate [mm/d]
input_climatology.delta_t  = 1;                                 % time step size of the inputs: 1 [d]                                                % time step size of the inputs: 1 [d]

% Extract observed streamflow
Q_obs = data.streamflow; % Daily data: Q rate [mm/d]

% Create an exploratory plot
figure;
subplot(211)
hold on;
plot(data.date,80-input_climatology.precip, 'b')
plot(data.date,Q_obs,'k')
legend('Precipitation','Streamflow')

subplot(212)
hold on;
plot(data.date,input_climatology.temp)
plot(data.date,input_climatology.pet)
legend('Temperature','Potential evapotranspiration')

disp(["Mean P  : ",mean(input_climatology.precip)])
disp(["Mean PET: ",mean(input_climatology.pet)])

%% 2. Define the model settings and create the model object
% We know we need snow here, so it seems reasonable to choose snow models
models = ["m_06_alpine1_4p_2s","m_12_alpine2_6p_2s","m_30_mopex2_7p_5s","m_34_flexis_12p_5s","m_37_hbv_15p_5s"];
model_ids = ["m06", "m12", "m30", "m34", "m36"];

for i = 1:length(models)
    model = models(i);
    model_id = model_ids(i);

    % Get the model particulars
    m = feval(model);
    parRanges = m.parRanges;                                                   % Parameter ranges
    numParams = m.numParams;                                                   % Number of parameters
    numStores = m.numStores;                                                   % Number of stores
    input_s0  = zeros(numStores,1);                                            % Initial storages (see note in paragraph 5 on model warm-up)
    
    %% 3. Define the solver settings  
    input_solver_opts.resnorm_tolerance = 0.1;                                 % Root-finding convergence tolerance;
    input_solver_opts.resnorm_maxiter   = 6;                                   % Maximum number of re-runs
    
    %% 4. Define calibration settings
    % Settings for 'my_cmaes'
    % the opts struct is made up of two fields, the names are hardcoded, so
    % they cannot be changed:
    %    .sigma0:     initial value of sigma
    %    .cmaes_opts: struct of options for cmaes, see cmaes documentation
    %                 or type cmaes to see list of options and default values
    
    % starting sigma
    optim_opts.insigma = .3*(parRanges(:,2) - parRanges(:,1));                 % starting sigma (this is default, could have left it blank)
    
    % other options
    optim_opts.LBounds  = parRanges(:,1);                                      % lower bounds of parameters
    optim_opts.UBounds  = parRanges(:,2);                                      % upper bounds of parameters
    optim_opts.PopSize  = 4 + floor(3*log(numParams));                         % population size (default)
    optim_opts.TolX       = 1e-6 * min(optim_opts.insigma);                    % stopping criterion on changes to parameters 
    optim_opts.TolFun     = 1e-6;                                              % stopping criterion on changes to fitness function
    optim_opts.TolHistFun = 1e-6;                                              % stopping criterion on changes to fitness function
    optim_opts.SaveFilename      = 'cmaes_temp_benchmark_cal_cmaesvars.mat';   % output file of cmaes variables
    optim_opts.LogFilenamePrefix = 'cmaes_temp_benchmark_cal_';                % prefix for cmaes log-files
    optim_opts.EvalParallel = true;                                            % change to false to disable run in parallel on a pool of CPUs (e.g. on a cluster)
    
    % initial parameter set
    par_ini = mean(parRanges,2);                                               % same as default value
    
    % Choose the objective function
    of_name      = 'of_KGE';                                                   % This function is provided as part of MARRMoT. See ./MARRMoT/Functions/Objective functions
    weights      = [1,1,1];                                                    % Weights for the three KGE components
    
    % Time periods for calibration (duplicate from Python)
    flow_mask = ~isnan(Q_obs);
    flow_dates = dates(flow_mask);
    median_date_id = idivide(length(flow_dates), int16(2), 'floor'); % mimic Python's // operator
    median_date = flow_dates(median_date_id); 
    assert(length(flow_dates) > 4*365, 'Less than 4 years of flow data found.')
    cal_mask = (flow_dates <= median_date);
    val_mask = ~cal_mask;
    cal_idx = find(cal_mask);
    val_idx = find(val_mask);
    
    %% 5. Calibrate the model
    % MARRMoT model objects have a "calibrate" method that takes uses a chosen
    % optimisation algorithm and objective function to optimise the parameter
    % set. See MARRMoT_model class for details.
    
    % first set up the model
    m.input_climate = input_climatology;
    m.solver_opts   = input_solver_opts;
    m.S0            = input_s0;
    
    % create output structures
    n_runs = 5; % number of calibration repeats
    q_sim_all = NaN*zeros(length(Q_obs),n_runs);
    of_cal_all = NaN*zeros(1,n_runs);
    of_val_all = NaN*zeros(1,n_runs);
    
    % Then run several repeats of the calibration
    for iRun = 1:n_runs
    
        % Change the optimizer seed
        optim_opts.Seed = iRun;                                                % for reproducibility
       
        % Optimize parameters
        [par_opt,...                                                           % optimal parameter set
        of_cal,...                                                             % value of objective function at par_opt
        stopflag,...                                                           % flag indicating reason the algorithm stopped
        output] = ...                                                          % other info about parametrisation
                  m.calibrate(...                                              % call the calibrate method of the model object
                              Q_obs,...                                        % observed streamflow
                              cal_idx,...                                      % timesteps to use for model calibration
                              'my_cmaes',...                                   % function to use for optimisation (must have same structure as fminsearch)
                              par_ini,...                                      % initial parameter estimates
                              optim_opts,...                                   % options to optim_fun
                              of_name,...                                      % name of objective function to use
                              1,1,...                                          % should the OF be inversed?   Should I display details about the calibration?
                              weights);                                        % additional arguments to of_name
    
        % Run the model with calibrated parameters, get only the streamflow
        Q_sim = m.get_streamflow([],[],par_opt);  
                     
        % Compute evaluation performance
        of_val = feval(of_name,...                                             % Objective function name (here 'of_KGE')
                       Q_obs,...                                               % Observed flow during evaluation period
                       Q_sim,...                                               % Simulated flow during evaluation period, using calibrated parameters            
                       val_idx,...                                             % Indices of evaluation period
                       weights);                                               % KGE component weights
    
        % Store the outputs
        q_sim_all(:,iRun) = Q_sim;
        of_cal_all(iRun) = of_cal;
        of_val_all(iRun) = of_val;
           
    end
    
    % Save the calibration outcomes to the simulation folder
    csv_table = [table(dates),array2table(q_sim_all, 'VariableNames',strcat('parset',string(1:n_runs)))];
    currentDateTime = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
    filename = join([output_folder,currentDateTime,'_',model_id,'_qsim.csv'],'');
    writetable(csv_table, filename);
    
    filename = join([output_folder,currentDateTime,'_',model_id,'_cal_kge.csv'],'');
    writematrix(of_cal_all, filename)
    
    filename = join([output_folder,currentDateTime,'_',model_id,'_val_kge.csv'],'');
    writematrix(of_val_all, filename)
    
    % Remove all the CMAES temporary files
    files_to_delete = dir(join([optim_opts.LogFilenamePrefix,'*']));
    files_to_delete(end+1) = dir(optim_opts.SaveFilename);
    for i = 1:length(files_to_delete)
        filepath = fullfile(files_to_delete(i).folder, files_to_delete(i).name); % Construct the full file path
        delete(filepath); % Delete the file
        fprintf('Deleted file: %s\n', filepath);
    end
end
