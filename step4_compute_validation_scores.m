function step4_compute_validation_scores(class)
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));
	
  load(sprintf('%s/hog_validation_set_for_learning_block_classifiers.mat', conf.dataDir));

  all_models = dir(sprintf('%s/%s/%d',conf.dataDir, model.path, class));
  all_models = all_models(3:end);

	system(sprintf('mkdir -p %s/comparison-classifiers-thresholding/%d',conf.dataDir, class));

  for model_idx = 1:size(all_models, 1)
    fprintf('%d\n',model_idx);
    model_file = all_models(model_idx).name;
    model_id = str2num(strtok(model_file,'.'));

    if(exist(sprintf('%s/comparison-classifiers-thresholding/%d/%d-validation-scores.mat',conf.dataDir, class,model_id)))
      continue
    end
		load(sprintf('%s/%s/%d/%d.mat',conf.dataDir, model.path, class, model_id));
		testSet = find(imdb.images.isTrain==1 & imdb.images.isTrainVal==0);
		tot_frames = {};
	  tot_gt = {};
	  tot_scores = zeros(length(testSet),100);
		for i=1:length(testSet)
			ii = testSet(i);
		  id = ii;
	    gt = imdb.images.class(ii); 
		  hog = tot_hog{i};
	    grad= tot_grad{i};
		  w = model.w ;
	  	w = single(reshape(w,model.height,model.width,[])) ;
	 	  f = zeros(4,0) ;
	    n = 1;
	    temp_scores = [];
		  for s = 1:numel(hog)
		    if(size(hog{s},1) < 8 || size(hog{s},2)<8)
		        continue
		    end
		    scores = vl_fconv(hog{s}, w) ;
		    [~,perm] = sort(scores(:)', 'descend') ;
			  %%%%%%%%%%% Discard low strength magnitudes blocks
			    tmp_idx = find(grad{s} == 0);
			    tmp_idx = find(ismember(perm,tmp_idx));
			    perm(tmp_idx) = [];
			  %%%%%%%%%%% Discard low strength ends  
		    n = min(100, length(perm));
		    [perm,sel] = vl_colsubset(perm, n, 'beginning') ;
	      temp_scores = [temp_scores scores(perm)];
		  end
	    temp_scores = sort(temp_scores, 'descend');
	    tot_scores(i,:) = temp_scores(1:100);
		end
	  save(sprintf('%s/comparison-classifiers-thresholding/%d/%d-validation-scores.mat',conf.dataDir, class,model_id),'tot_scores');
  end

end
