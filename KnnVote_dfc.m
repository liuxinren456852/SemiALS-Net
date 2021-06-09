%%% Knn Vote for the orginal sets, i.e., 411722

mkdir('./data/dfc_data/Inference/kNN_out/Eval_CLS_weak_pointsift_v2/');

pred_root = './data/dfc_data/Inference/Eval_can_out_weak_pointsift_v2/';
eval_files = dir([pred_root, '*.txt']);

for i=1:10
    eval_f = eval_files(i).name

    can_path=fullfile([pred_root, eval_f]); 
    out_filepath=fullfile(['./data/dfc_data/Inference/kNN_out/Eval_CLS_weak_pointsift_v2/', strrep(eval_f,'can','CLS')]); 
    
    pc_f = strrep(eval_f,'can','PC3');
    pc3_path=fullfile(['./data/dfc_data/Track4/', pc_f]);
    
    src_file = fullfile(['./data/dfc_data/Track4-Truth/', strrep(eval_f,'can','CLS')]);
    dst_file = fullfile(['./data/dfc_data/Inference/gt_test/', strrep(eval_f,'can','CLS')]);
    copyfile(src_file, dst_file);

    out=fopen(out_filepath,'w');
    ca=dlmread(can_path,','); %candidate points
    pc=dlmread(pc3_path, ','); %test points

    pset.x=pc(:,1);
    pset.y=pc(:,2); 
    pset.z=pc(:,3);
    [r,c]=size(pc)
    % pc_xyz size: n*3

    pc_xyz=zeros(r,3);
    pc_xyz(:,1)=pset.x;
    pc_xyz(:,2)=pset.y;
    pc_xyz(:,3)=pset.z;

    cset.x=ca(:,1);
    cset.y=ca(:,2); 
    cset.z=ca(:,3);
    cset.c=ca(:,6); 
    [rc,cc]=size(ca)
    % pc_xyz size: n*3

    c_xyz=zeros(rc,3);
    c_xyz(:,1)=cset.x;
    c_xyz(:,2)=cset.y;
    c_xyz(:,3)=cset.z;

    % finds the nearest neighbor in X for each point in Y
    tic;[idx, dist] = knnsearch(c_xyz,pc_xyz,'dist','euclidean','k',24);toc
    label_mat = [0,2,5,6,9,17];
    tic;
    for i = 1:r
        count=zeros(1,6);
        
        for j = 1:24
            num=idx(i,j);
            if cset.c(num)==label_mat(1)
                count(1)=count(1)+1;
            elseif cset.c(num)==label_mat(2)
                count(2)=count(2)+1;
            elseif cset.c(num)==label_mat(3)
                count(3)=count(3)+1;
            elseif cset.c(num)==label_mat(4)
                count(4)=count(4)+1;
            elseif cset.c(num)==label_mat(5)
                count(5)=count(5)+1;
            elseif cset.c(num)==label_mat(6)
                count(6)=count(6)+1;
            end
        end

        [val,index]=max(count); 
        new_label=label_mat(index);
        fprintf(out,'%d\n',new_label);
    end
    toc;
    fclose(out);

end
