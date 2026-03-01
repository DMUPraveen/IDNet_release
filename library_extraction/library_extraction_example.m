pkg load statistics

% 1. Get the command line arguments
args = argv();

% Ensure the user actually provided a dataset name
if (length(args) < 1)
  error('Missing dataset name. Usage: octave process_dataset.m <dataset_name>');
end

% 2. Extract the dataset name (first argument)
dataset_name = args{1};
% 3. Construct the input and output filenames

input_filename = sprintf('%s.mat', dataset_name);
output_filename = sprintf('../DATA/%s_bundles.mat', dataset_name);

% // % load Yim : nr * nc * L  hyperspectral image cube
% // load '../DATA/real_Samson/alldata_real_Samson.mat'
addpath('../DATA/');
load(input_filename);
Y = reshape(Yim, [size(Yim,1)*size(Yim,2), size(Yim,3)])'; % reorder into bands * pixels

num_EMs = size(M0)(2);

% extract reference EMs using VCA if desired
% M0 = vca(Y,'Endmembers',num_EMs); 


% Extract bundles from the image based on angles (see the description in the IDNet paper) -----------
flag_Npx = true;
vec_Npx = 100*ones(num_EMs,1); % number of pure pixel to extract per endmember
[bundleLibs,avg_M,PPidx,EM_pix,IDX_comp] = extract_bundles_by_angle(Yim, M0, vec_Npx, flag_Npx);


% Extract bundles from the image based on the batch VCA alg. -----------
percent = 0.25;
bundle_nbr = 100;
[bundleLibs2]=extractbundles_batchVCA(Y, M0, bundle_nbr, percent);


save(output_filename,'bundleLibs')


