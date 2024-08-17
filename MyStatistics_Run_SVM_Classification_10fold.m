function [Results] = MyStatistics_Run_SVM_Classification_10fold(Features, Group_label, N_repe, Method, BoxConstraint)

%==========================================================================
% Contributor:
% Yuping Yang, UoM, Manchester, yuping.yang@postgrad.manchester.ac.uk
% Junle Li, IBRR, SCNU, GuangZhou, lijunle.1995@gmail.com
% Jinhui Wang, IBRR, SCNU, GuangZhou, jinhui.wang.1982@gmail.com
% Anna Woollams, UoM, Manchester, anna.woollams@manchester.ac.uk
% Nelson Trujillo-Barreto, UoM, Manchester, nelson.trujillo-barreto@manchester.ac.uk
% Nils Muhlert, UoM, Manchester, nils.muhlert@manchester.ac.uk
%==========================================================================

if nargin < 3
    error('At least three arguments are required!')
elseif nargin == 3
    Method = 'both';
    BoxConstraint = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
elseif nargin == 4
    BoxConstraint = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
end

K_fold = 10;
FeatureReduction = {'No','PCA'};
Results = struct;
Results(1) = [];

if strcmpi(Method,'SVM') || strcmpi(Method,'both')
    % ==== SVM ====
    SVM_Kernel = {'Linear','Gaussian','Polynomial'};
    for boxConstraint = BoxConstraint
        for svm_kernel = SVM_Kernel
            if strcmpi(svm_kernel,'Polynomial')
                poly_order = 2;
            else
                poly_order = [];
            end
            for featureReduction = FeatureReduction
                disp(['Performing SVM classification - BoxConstraint(', int2str(boxConstraint),') - SVM_Kernel(',svm_kernel{1},') - FeatureReduction(', featureReduction{1}, ') ... | ',datestr(clock)]);
                Results(end+1).Method = 'SVM';
                Results(end).SVM_Kernel = svm_kernel{1};
                Results(end).SVM_BoxConstraint = boxConstraint;
                Results(end).FeatureReduction = featureReduction{1};
                acc_sen_spe = zeros(N_repe,3);
                for i_repe = 1:N_repe
                    results = MyStatistics_classification_2groups_SVM_Kfold(Features, Group_label, K_fold, svm_kernel{1}, boxConstraint, poly_order, featureReduction{1});
                    acc_sen_spe(i_repe,1) = results.Accuracy;
                    acc_sen_spe(i_repe,2) = results.Sensitivity;
                    acc_sen_spe(i_repe,3) = results.Specificity;
                end
                Results(end).Accuracy = mean(acc_sen_spe(:,1));
                Results(end).Sensitivity = mean(acc_sen_spe(:,2));
                Results(end).Specificity = mean(acc_sen_spe(:,3));
                disp(['SVM classification - BoxConstraint(', int2str(boxConstraint),') - SVM_Kernel(',svm_kernel{1},') - FeatureReduction(', featureReduction{1}, ') is done ..................... | ',datestr(clock)]);
            end
        end
    end
end

return
