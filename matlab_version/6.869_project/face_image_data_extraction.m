clear
addpath('../PDM_helpers/');
addpath(genpath('../fitting/'));
addpath('../models/');
addpath(genpath('../face_detection'));
addpath('../CCNF/');

%% loading the patch experts
   
[clmParams, pdm] = Load_CLM_params_wild();

% An accurate CCNF (or CLNF) model
[patches] = Load_Patch_Experts( '../models/general/', 'ccnf_patches_*_general.mat', [], [], clmParams);
% A simpler (but less accurate SVR)
% [patches] = Load_Patch_Experts( '../models/general/', 'svr_patches_*_general.mat', [], [], clmParams);

clmParams.multi_modal_types  = patches(1).multi_modal_types;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%need to direct to the dataset of our porject
%imagesRootDir is the root directory of the KDEF dataset
imagesRootPath='../../../../../Dropbox/Study//6.869/6.869 Project/Dataset/KDEF';
imagesRootDir = dir(imagesRootPath);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
isub = [imagesRootDir(:).isdir]; %# returns logical vector of whether it is a subfolder
imagesSubFolders = {imagesRootDir(isub).name}';%only keeps subfolder
imagesSubFolders(ismember(imagesSubFolders,{'.','..'})) = [];%remove the useless '.','..'

%%
verbose = true;

for fod=1:numel(imagesSubFolders)
% for fod =1:1:1;
    currentImageFolder=[imagesRootPath,'/',imagesSubFolders{fod}];
    currentImageBundle=dir([currentImageFolder,'/*S.JPG']);%S. is for straight (front) face
%     currentImageBundle=dir([currentImageFolder,'/*H?.JPG']);%H?. is for half left or half right face
    
    for img=1:numel(currentImageBundle);
%     for img=1:1:1;
    currentImage=[currentImageFolder,'/', currentImageBundle(img).name];
    imageOrig = imread(currentImage);

    % First attempt to use the Matlab one (fastest but not as accurate, if not present use yu et al.)
    [bboxs, det_shapes] = detect_faces(imageOrig, {'cascade', 'yu'});
    % Zhu and Ramanan and Yu et al. are slower, but also more accurate 
    % and can be used when vision toolbox is unavailable
    % [bboxs, det_shapes] = detect_faces(imageOrig, {'yu', 'zhu'});
    
    % The complete set that tries all three detectors starting with fastest
    % and moving onto slower ones if fastest can't detect anything
    % [bboxs, det_shapes] = detect_faces(imageOrig, {'cascade', 'yu', 'zhu'});
    
    if(size(imageOrig,3) == 3)
        image = rgb2gray(imageOrig);
    end              

    

    if(verbose)
        f = figure(1);clf;    
        if(max(image(:)) > 1)
            imshow(double(imageOrig)/255, 'Border', 'tight');
        else
            imshow(double(imageOrig), 'Border', 'tight');
        end
        axis equal;
        hold on;
    end

    for i=1:size(bboxs,2)

        % Convert from the initial detected shape to CLM model parameters,
        % if shape is available
        
        bbox = bboxs(:,i);
        
        if(~isempty(det_shapes))
            shape = det_shapes(:,:,i);
            inds = [1:60,62:64,66:68];
            M = pdm.M([inds, inds+68, inds+68*2]);
            E = pdm.E;
            V = pdm.V([inds, inds+68, inds+68*2],:);
            [ a, R, T, ~, params, err, shapeOrtho] = fit_PDM_ortho_proj_to_2D(M, E, V, shape);
            g_param = [a; Rot2Euler(R)'; T];
            l_param = params;

            % Use the initial global and local params for clm fitting in the image
            [shape,~,~,lhood,lmark_lhood,view_used] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams, 'gparam', g_param, 'lparam', l_param);
        else
            [shape,~,~,lhood,lmark_lhood,view_used] = Fitting_from_bb(image, [], bbox, pdm, patches, clmParams);
        end
        
        % shape correction for matlab format
        shape = shape + 1;

        if(verbose)

            % valid points to draw (not to draw self-occluded ones)
            v_points = logical(patches(1).visibilities(view_used,:));

            try

            plot(shape(v_points,1), shape(v_points',2),'.r','MarkerSize',20);
            plot(shape(v_points,1), shape(v_points',2),'.b','MarkerSize',10);

            catch warn

            end
        end

    end
    hold off;
    [~,imageName,~] = fileparts(currentImage);
    save([currentImageFolder,'/',imageName,'.mat'],'shape','v_points','lhood','lmark_lhood','view_used','bboxs','det_shapes');
    end
end