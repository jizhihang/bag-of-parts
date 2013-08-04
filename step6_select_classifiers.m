function step6_select_classifiers
  load_vlfeat('0.9.16');
  config;

  selected_classifiers = {};

  num_classifiers = 100;

  for i=1:conf.numClasses

    load(sprintf('%s/%s/fin-entropy/%d.mat', conf.dataDir, conf.entropyDir, i));

    tot_models = length(tot_AP);
  
    entropy_rank_auc = tot_entropy_auc;

    [c, perm] = sort(entropy_rank_auc, 'ascend');

    rank_entropy = zeros(tot_models,1);
    rank_entropy(perm) = [1:tot_models];
    combined_rank = rank_entropy;
    [c,perm] = sort(combined_rank, 'ascend');

    selected_classifiers_cur = [perm(1)];
    cur = 2;

    while(length(selected_classifiers_cur) < num_classifiers)
      flag = 1;
      cur_w = load(sprintf('%s/%s/%d/%d.mat', conf.dataDir, model.path, i, perm(cur)));
      cur_w = cur_w.model.w;
      for j=1:length(selected_classifiers_cur)
        test_w = load(sprintf('%s/%s/%d/%d.mat', conf.dataDir, model.path, i, selected_classifiers_cur(j)));
        test_w = test_w.model.w;
        similarity = (cur_w' * test_w) / (norm(cur_w) * norm(test_w));
        if(similarity > 0.5)
          flag = 0;
          break;
        end
      end
      if(flag == 1)
        selected_classifiers_cur = [selected_classifiers_cur perm(cur)];
      end
      cur = cur+1;
    end
    selected_classifiers{i} = selected_classifiers_cur;
  end 
  save(sprintf('%s/%s/selected_classifiers.mat', conf.dataDir, conf.entropyDir),'selected_classifiers');
end
