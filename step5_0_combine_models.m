function step5_0_combine_models
  load_vlfeat('0.9.16');
  config;
  % First, combine all the models for each class
  for class = 1:conf.numClasses

	  all_models = dir(sprintf('%s/%s/%d', conf.dataDir, model.path, class));
	  all_models = all_models(3:end);
	
	  num_models = length(all_models);
	  combined_w = zeros(model.length, num_models, 'single');
	  
	  for jj=1:num_models
	    load(sprintf('%s/%s/%d/%d.mat', conf.dataDir, model.path, class, jj));
	    combined_w(:,jj) = model.w;
	  end
	  save(sprintf('%s/%s/combined-models-%d.mat', conf.dataDir, model.path, class),  'combined_w', '-v7.3');
  end

  % Second, combine all the models of all the classes
  combined_w = [];
  for class=1:67
    a=load(sprintf('%s/%s/combined-models-%d.mat', conf.dataDir, model.path, class));
    combined_w = [combined_w a.combined_w];
    class
  end
  save(sprintf('%s/%s/combined-models.mat', conf.dataDir, model.path),'combined_w','-v7.3');
end
