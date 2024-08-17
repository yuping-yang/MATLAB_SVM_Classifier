function Results = MyStatistics_SVM_Classification_Train_Test(Features_train, ClassLabel_train, Features_test, ClassLabel_test, Kernel, C, Poly_order, C_type, C_thr)

%==========================================================================
% Binary classification based on support vector machine models.
% Contributor:
% Yuping Yang, UoM, Manchester, yuping.yang@postgrad.manchester.ac.uk
% Junle Li, IBRR, SCNU, GuangZhou, lijunle.1995@gmail.com
% Jinhui Wang, IBRR, SCNU, GuangZhou, jinhui.wang.1982@gmail.com
% Anna Woollams, UoM, Manchester, anna.woollams@manchester.ac.uk
% Nelson Trujillo-Barreto, UoM, Manchester, nelson.trujillo-barreto@manchester.ac.uk
% Nils Muhlert, UoM, Manchester, nils.muhlert@manchester.ac.uk
%==========================================================================

if nargin < 4
    error('At least 4 arguments are required!');
end

if nargin == 4
    Kernel = 'Linear';
    C = 1;
    Poly_order = [];
    C_type = 'No';
    C_thr = [];
end

if nargin == 5
    C = 1;
    if strcmpi(Kernel,'Polynomial')
        Poly_order = 2;
    else
        Poly_order = [];
    end
    C_type = 'No';
    C_thr = [];
end

if nargin == 6
    if strcmpi(Kernel,'Polynomial')
        Poly_order = 2;
    else
        Poly_order = [];
    end
    C_type = 'No';
    C_thr = [];
end

if nargin == 7
    C_type = 'No';
    C_thr = [];
end

if nargin == 8
    if strcmpi(C_type,'No')
        C_thr = [];
    elseif strcmpi(C_type,'PCA')
        C_thr = 80;
    elseif strcmpi(C_type,'Ttest')
        C_thr = 0.05;
    end
end

if nargin > 9
    error('At most 8 arguments are required!');
end

[Nsub_train, Nvar_train] = size(Features_train);
[Nsub_test, Nvar_test] = size(Features_test);

if length(ClassLabel_train) ~= Nsub_train || length(ClassLabel_test) ~= Nsub_test
    error('The number of observations are not equal between Features and Group_label!');
end

if length(unique(ClassLabel_train)) ~= 2 || length(unique(ClassLabel_test)) ~= 2
    error('The number of classes must be 2!');
end

if length(Nvar_train) ~= length(Nvar_test)
    error('The number of variables are not equal between the training and testing dataset!');
end

if ~(strcmpi(Kernel,'Gaussian') || strcmpi(Kernel,'Linear') || strcmpi(Kernel,'Polynomial'))
    error('Unrecognized input for Kernel!');
end

if C <= 0
    error('C should be a positive scalar!');
end

if strcmpi(Kernel,'Polynomial') && (Poly_order < 0 || mod(Poly_order,1) ~= 0)
    error('Poly_order for polynomial kernel should be a positive integer!');
end

if ~(strcmpi(C_type,'No') || strcmpi(C_type,'PCA') || strcmpi(C_type,'Ttest'))
    error('Unrecognized input for C_type!');
end

if strcmpi(C_type,'PCA')
    if C_thr <= 0 || C_thr > 100
        error('The range of C_type for PCA should be (0 100]!');
    end
end

if strcmpi(C_type,'Ttest')
    if C_thr <= 0 || C_thr >= 1
        error('The range of C_type for T_test should be (0 1)!');
    end
end

Beta = zeros(1, Nvar_train);

if strcmpi(C_type,'PCA')
    Explained_variance = 0;
end
if strcmpi(C_type,'Ttest')
    Feature_sig = zeros(1, Nvar_train);
end

% Compensating the imbalance between classes 
% Set the weight for each class based on the number of instances within that class
ClassWeight_train = zeros(size(ClassLabel_train));
for i_label = unique(ClassLabel_train)'
    ClassWeight_train(ClassLabel_train==i_label) = 1/(sum(ClassLabel_train==i_label));
end  

ClassWeight_test = zeros(size(ClassLabel_test));
for i_label = unique(ClassLabel_test)'
    ClassWeight_test(ClassLabel_test==i_label) = 1/(sum(ClassLabel_test==i_label));
end

% selection to feature
if strcmpi(C_type,'PCA')
    [Coeff_train, Score_train, ~, ~, Explained_train, mu] = pca(Features_train);
    Score_test = (Features_test - repmat(mu,size(Features_test,1),1)) * Coeff_train;
    sum_Explained_train = cumsum(Explained_train);
    Ind_component = find(sum_Explained_train >= C_thr,1);
    Explained_variance = sum_Explained_train(Ind_component);
    Features_train = Score_train(:,1:Ind_component);
    Features_test = Score_test(:,1:Ind_component);
end

if strcmpi(C_type,'Ttest')
    [~,p] = ttest2(Features_train(ClassLabel_train == 0,:),Features_train(ClassLabel_train == 1,:));
    Features_train = Features_train(:,p <= C_thr);
    Features_test = Features_test(:,p <= C_thr);
    Feature_sig(1, p <= C_thr) = 1;
    if sum(Feature_sig) == 0
        error('C_thr is too strict for feature selection!');
    end
end

% Standardization
mean_Features_train = mean(Features_train);
std_Features_train = std(Features_train);
Zscore_Features_train = (Features_train - mean_Features_train)./std_Features_train;
Zscore_Features_test = (Features_test - mean_Features_train)./std_Features_train; 

% Model construction
if strcmpi(Kernel,'Linear')
    Mdl = fitcsvm(Zscore_Features_train,ClassLabel_train,'BoxConstraint',C,'Standardize',0,'Weights',ClassWeight_train);
    if strcmpi(C_type,'no')
        Beta = Mdl.Beta;
    elseif strcmpi(C_type,'PCA')
        Beta = Coeff_train(:,1:Ind_component) * Mdl.Beta;
    else
        Beta(1,logical(Feature_sig)) = Mdl.Beta;
    end
elseif strcmpi(Kernel,'Gaussian')
    Mdl = fitcsvm(Zscore_Features_train,ClassLabel_train,'KernelFunction','gaussian','BoxConstraint',C,'Standardize',0,'Weights',ClassWeight_train);
elseif strcmpi(Kernel,'Polynomial')
    Mdl = fitcsvm(Zscore_Features_train,ClassLabel_train,'KernelFunction','Polynomial',...
        'BoxConstraint',C,'PolynomialOrder',Poly_order,'Standardize',0,'Weights',ClassWeight_train);
end

ClassLabel_predict = predict(Mdl,Zscore_Features_test);

Results_matrix= zeros(2);
Results_matrix(1,1) = sum(ClassLabel_test == 0 & ClassLabel_predict == 0);
Results_matrix(1,2) = sum(ClassLabel_test == 0 & ClassLabel_predict == 1);
Results_matrix(2,1) = sum(ClassLabel_test == 1 & ClassLabel_predict == 0);
Results_matrix(2,2) = sum(ClassLabel_test == 1 & ClassLabel_predict == 1);

Results = struct;
Results.ClassLabel_predicted = ClassLabel_predict;
Results.Matrix = Results_matrix;
Results.Accuracy = (Results_matrix(1,1) + Results_matrix(2,2)) / Nsub_train;
Results.Sensitivity = Results_matrix(2,2) / (Results_matrix(2,1) + Results_matrix(2,2));
Results.Specificity = Results_matrix(1,1) / (Results_matrix(1,1) + Results_matrix(1,2));
Results.Kernel = Kernel;
Results.C = C;
if strcmpi(Kernel,'Polynomial')
    Results.Poly_order = Poly_order;
end
Results.C_type = C_type;
Results.C_thr = C_thr;
if strcmpi(Kernel,'Linear')
    Results.Beta = Beta;
end
if strcmpi(C_type,'PCA')
    Results.Explained_variance = Explained_variance;
end

return
