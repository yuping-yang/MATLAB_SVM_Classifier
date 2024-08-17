function [Results,ROC_x_y] = MyStatistics_SVM_ClassificationRepeat_Kfold(Features, Group_label, Output_dir, Output_filename, K_fold, SVM_kernel, BoxConstraint, Poly_order, FeatureReduction, N_repe)

%==========================================================================
% Contributor:
% Yuping Yang, UoM, Manchester, yuping.yang@postgrad.manchester.ac.uk
% Junle Li, IBRR, SCNU, GuangZhou, lijunle.1995@gmail.com
% Jinhui Wang, IBRR, SCNU, GuangZhou, jinhui.wang.1982@gmail.com
% Anna Woollams, UoM, Manchester, anna.woollams@manchester.ac.uk
% Nelson Trujillo-Barreto, UoM, Manchester, nelson.trujillo-barreto@manchester.ac.uk
% Nils Muhlert, UoM, Manchester, nils.muhlert@manchester.ac.uk
%==========================================================================

% ==== Input arguments ====
if nargin > 10
    error('At most 10 arguments are required!')
end
if nargin < 10
    N_repe = 100;
end
if nargin < 9
    FeatureReduction = 'no';
end
if nargin < 8
    Poly_order = [];
end
if nargin < 7
    BoxConstraint = 1;
end
if nargin < 6
    SVM_kernel = 'linear';
end
if nargin < 5
    K_fold = 10;
end
if nargin < 4
    error('At least 4 arguments are required!')
end

% ==== Classification ====
Results = struct;
Results(1) = [];
ROC_x_y = zeros(length(Group_label)+1,2,N_repe); % (N+1)*2 matrix indicating the X and Y coordinates of the ROC curve.
for i_repe = 1:N_repe
    Results(end+1).Method = 'SVM';
    Results(end).SVM_Kernel = SVM_kernel;
    Results(end).SVM_BoxConstraint = BoxConstraint;
    Results(end).FeatureReduction = FeatureReduction;
    result = MyStatistics_classification_2groups_SVM_Kfold(Features, Group_label, K_fold, SVM_kernel, BoxConstraint, Poly_order, FeatureReduction);
    Results(end).Accuracy = result.Accuracy;
    Results(end).Sensitivity = result.Sensitivity;
    Results(end).Specificity = result.Specificity;
    Results(end).Precision = result.Precision;
    Results(end).AUC = result.AUC;
    ROC_x_y(:,:,i_repe) = result.ROC_x_y;
end
Results = [Results(1), Results];
Results(1).Method = 'Mean of N_repe';
Results(1).SVM_Kernel = 'Mean of N_repe';
Results(1).SVM_BoxConstraint = 'Mean of N_repe';
Results(1).FeatureReduction = 'Mean of N_repe';
Results(1).Accuracy = mean([Results(2:end).Accuracy]);
Results(1).Sensitivity = mean([Results(2:end).Sensitivity]);
Results(1).Specificity = mean([Results(2:end).Specificity]);
Results(1).Precision = mean([Results(2:end).Precision]);
Results(1).AUC = mean([Results(2:end).AUC]);
ROC_x_y = mean(ROC_x_y,3);
disp(['SVM classification - BoxConstraint(', int2str(BoxConstraint),') - SVM_Kernel(',SVM_kernel,') - FeatureReduction(', FeatureReduction, ') is done ..................... | ',datestr(clock)]);
cd(Output_dir)
eval([Output_filename,' = Results;'])
save([Output_filename,'.mat'],Output_filename,'ROC_x_y')
disp(['Completed: SVM classification - BoxConstraint(', int2str(BoxConstraint),') - SVM_Kernel(',SVM_kernel,') - FeatureReduction(', FeatureReduction, '):'])
disp(['========== Results: Mean Accuracy = ',num2str(Results(1).Accuracy,'%.3f'),' & Mean Sensitivity = ',num2str(Results(1).Sensitivity,'%.3f'),' & Mean Specificity = ',num2str(Results(1).Specificity,'%.3f'),' & Mean Precision = ',num2str(Results(1).Precision,'%.3f'),' & Mean AUC = ',num2str(Results(1).AUC,'%.3f'),' ==========']);

end
