function [fpr, tpr, auc_score] = calc_roc_curve(output, label)

% Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for different thresholds
thresholds = 0:0.01:1;
tpr = zeros(length(thresholds),1);
fpr = zeros(length(thresholds),1);

for i = 1:length(thresholds)
    % Binary classification based on threshold
    binary_output = output >= thresholds(i);
    
    % Calculate True Positive and False Positive counts
    tp_count = sum(binary_output(label==1)==1);
    fp_count = sum(binary_output(label==0)==1);
    
    % Calculate TPR and FPR
    tpr(i) = tp_count / sum(label==1);
    fpr(i) = fp_count / sum(label==0);
end

% Calculate AUC
auc_score = trapz(fpr,tpr);

% Plot ROC curve
% figure;
% plot(fpr, tpr); 
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
% title('ROC Curve');

end



