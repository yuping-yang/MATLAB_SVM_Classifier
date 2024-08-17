function Results = MyStatistics_SVM_Classification_Kfold(Features, Group_label, K_fold, Kernel, C, Poly_order, C_type, C_thr)

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

if nargin < 3
    error('At least 3 arguments are required!');
end

if nargin == 3
    Kernel = 'Linear';
    C = 1;
    Poly_order = [];
    C_type = 'No';
    C_thr = [];
end

if nargin == 4
    C = 1;
    if strcmpi(Kernel,'Polynomial')
        Poly_order = 2;
    else
        Poly_order = [];
    end
    C_type = 'No';
    C_thr = [];
end

if nargin == 5
    if strcmpi(Kernel,'Polynomial')
        Poly_order = 2;
    else
        Poly_order = [];
    end
    C_type = 'No';
    C_thr = [];
end

if nargin == 6
    C_type = 'No';
    C_thr = [];
end

if nargin == 7
    if strcmpi(C_type,'No')
        C_thr = [];
    elseif strcmpi(C_type,'PCA')
        C_thr = 80;
    elseif strcmpi(C_type,'Ttest')
        C_thr = 0.05;
    end
end

if nargin > 8
    error('At most 8 arguments are required!');
end

[Nsub, Nvar] = size(Features);

if length(Group_label) ~= Nsub
    error('The number of observations are not equal between Features and Group_label!');
end

if length(unique(Group_label)) ~= 2
    error('The number of Group_label type must be 2!');
end

if K_fold <= 1 || mod(K_fold,1) ~= 0
    error('K_fold must be a >1 positive integer!');
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

ClassLabel_predict = zeros(Nsub,1);
Scores = zeros(Nsub,2);

Feature_fold = zeros(Nvar,K_fold);
Beta_fold = zeros(Nvar,K_fold);
if strcmpi(C_type,'PCA')
    Explained_variance = zeros(K_fold,1);
end

Cvpar = cvpartition(Group_label,'KFold',K_fold);
for ifold = 1:K_fold
    disp(['Classifying the ' num2str(ifold) 'th fold |' datestr(clock)])
    Idx_train = Cvpar.training(ifold);
    Idx_test = ~Idx_train;
    Features_train = Features(Idx_train,:);
    ClassLabel_train = Group_label(Idx_train);
    Features_test = Features(Idx_test,:);

    % Compensating the imbalance between classes
    % Set the weight for each class based on the number of instances within that class
    ClassWeight_train = zeros(size(ClassLabel_train));
    for i_label = unique(ClassLabel_train)'
        ClassWeight_train(ClassLabel_train==i_label) = 1/(sum(ClassLabel_train==i_label));
    end

    % selection to feature
    if strcmpi(C_type,'No')
        Feature_fold(:,ifold) = 1;
    end

    if strcmpi(C_type,'PCA')
        [Coeff_train, Score_train, ~, ~, Explained_train, mu] = pca(Features_train);
        Score_test = (Features_test - repmat(mu,size(Features_test,1),1)) * Coeff_train;
        sum_Explained_train = cumsum(Explained_train);
        Ind_component = find(sum_Explained_train >= C_thr,1);
        Explained_variance(ifold) = sum_Explained_train(Ind_component);
        Features_train = Score_train(:,1:Ind_component);
        Features_test = Score_test(:,1:Ind_component);
        Feature_fold(:,ifold) = 1;
    end

    if strcmpi(C_type,'Ttest')
        [~,p] = ttest2(Features_train(ClassLabel_train == 0,:),Features_train(ClassLabel_train == 1,:));
        Features_train = Features_train(:,p <= C_thr);
        Features_test = Features_test(:,p <= C_thr);
        Feature_fold(p <= C_thr,ifold) = 1;
        if sum(Feature_fold(:,ifold)) == 0
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
        if strcmpi(C_type,'PCA')
            Beta_fold(:,ifold) = Coeff_train(:,1:Ind_component) * Mdl.Beta;
        else
            Beta_fold(logical(Feature_fold(:,ifold)),ifold) = Mdl.Beta;
        end
    elseif strcmpi(Kernel,'Gaussian')
        Mdl = fitcsvm(Zscore_Features_train,ClassLabel_train,'KernelFunction','gaussian','BoxConstraint',C,'Standardize',0,'Weights',ClassWeight_train);
    elseif strcmpi(Kernel,'Polynomial')

        Mdl = fitcsvm(Zscore_Features_train,ClassLabel_train,'KernelFunction','Polynomial',...
            'BoxConstraint',C,'PolynomialOrder',Poly_order,'Standardize',0,'Weights',ClassWeight_train);
    end

    [ClassLabel_predict(Idx_test),Scores(Idx_test,:)] = predict(Mdl,Zscore_Features_test);
    disp(['Classifying the ' num2str(ifold) 'th fold ...... is done |' datestr(clock)])
end

[ROC_x,ROC_y,~,AUC] = perfcurve(Group_label,Scores(:,2),1);

Results_matrix = zeros(2);
Results_matrix(1,1) = sum(Group_label == 1 & ClassLabel_predict == 1);
Results_matrix(1,2) = sum(Group_label == 1 & ClassLabel_predict == 0);
Results_matrix(2,1) = sum(Group_label == 0 & ClassLabel_predict == 1);
Results_matrix(2,2) = sum(Group_label == 0 & ClassLabel_predict == 0);

Results = struct;
Results.ClassLabel_predicted = ClassLabel_predict;
Results.Matrix = Results_matrix;
Results.Accuracy = (Results_matrix(1,1) + Results_matrix(2,2)) / Nsub;
Results.Sensitivity = Results_matrix(1,1) / (Results_matrix(1,1) + Results_matrix(1,2));
Results.Specificity = Results_matrix(2,2) / (Results_matrix(2,1) + Results_matrix(2,2));
Results.Precision = Results_matrix(1,1) / (Results_matrix(1,1) + Results_matrix(2,1));
Results.AUC = AUC;
Results.ROC_x_y = [ROC_x,ROC_y];
Results.Train_size = Cvpar.TrainSize;
Results.Kernel = Kernel;
Results.C = C;
if strcmpi(Kernel,'Polynomial')
    Results.Poly_order = Poly_order;
end
Results.C_type = C_type;
Results.C_thr = C_thr;
if strcmpi(C_type,'Ttest')
    Results.Consensus_numbers = sum(Feature_fold,2);
end
if strcmpi(Kernel,'Linear')
    Results.Beta_fold = Beta_fold;
    Results.Consensus_weights = sum(Beta_fold,2);
end
if strcmpi(C_type,'PCA')
    Results.Explained_variance = Explained_variance;
end

return
