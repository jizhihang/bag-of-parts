function step5_3_compute_entropy_final(class)
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  system(sprintf('mkdir -p %s/%s/fin-entropy', conf.dataDir, conf.entropyDir));

  class_scores = load(sprintf('%s/%s/%s/%d-scores.mat', conf.dataDir, conf.entropyDir, conf.finscoresDir, class));
  fin_scores = class_scores.fin_scores;
  ground_truth = class_scores.gt;
  num_classifiers = size(fin_scores,1);
  tot_entropy = zeros(size(fin_scores),'single'); 
  tot_entropy_auc = zeros(size(fin_scores,1),1,'single'); 
  tot_AP = zeros(size(fin_scores,1),1,'single'); 

  for i=1:num_classifiers
    fprintf('%d/%d\n', i,num_classifiers);
    [~,perm] = sort(fin_scores(i,:), 'ascend');
    top_classes = ground_truth(perm);
    entropy = calculate_entropy(top_classes, conf)';
    tot_entropy(i,:) = entropy;
    tot_entropy_auc(i) =0.5 * sum(entropy(1:end-1) + entropy(2:end)) ;
    
    labels = ground_truth;
    scores = fin_scores(i,:);
    labels = (labels==class)*2 - 1;
    [a,b,c] =  vl_pr(labels,scores);
    tot_AP(i) = c.auc;
  end

  save(sprintf('%s/%s/fin-entropy/%d.mat', conf.dataDir, conf.entropyDir, class),'tot_entropy','tot_entropy_auc','tot_AP');

end 

function entropy = calculate_entropy(top_classes, conf)
    orig_hist = single(zeros(length(top_classes), conf.numClasses));
    for i=1:conf.numClasses
        temp = cumsum(top_classes == i);
        orig_hist(:,i) = temp;
    end
    entropy = zeros(length(top_classes),1);
    for i=1:length(top_classes)
        hist = orig_hist(i, :);
        hist = bsxfun(@rdivide,hist,sum(hist,2));
        hist_log = log2(hist+eps);
        h_h_log = hist .* hist_log;
        clear hist hist_log;
        entropy(i) = -sum(h_h_log,2);
    end
    clear orig_hist
end
