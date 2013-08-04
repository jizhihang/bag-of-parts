function step5_2_compute_entropy_combine
  
  config;

  %%%%%%%%%%%%%%%%%%STEP 1: Combine top 5 detections of all models  from all validation images

  imdb = load(fullfile(conf.dataDir, conf.imdb));
  valSet = find(imdb.images.isTrain==1 & imdb.images.isTrainVal==0);
 
  combined_models = load(sprintf('%s/%s/combined-models.mat', conf.dataDir, model.path));
  num_models = size(combined_models.combined_w, 2);

  system(sprintf('mkdir -p %s/%s/%s', conf.dataDir, conf.entropyDir, conf.scoresDir));

  for run=1:5  
	  all_scores = dir(sprintf('%s/%s/top5detections', conf.dataDir, conf.entropyDir));
	  all_scores = all_scores(3:end);
	  num = length(all_scores);
	  range_ = [];
	  for i=0:4
	    range_ = [range_ num_models*i];
	  end
	  fin_selected = -999*ones(num_models, length(valSet), 'single'); 

	  for i=1:num
	    fprintf('%d\n',i);
	    cur_id = str2num(strtok(all_scores(i).name,'.'));
	    cur_scores = load(sprintf('%s/%s/top5detections/%s', conf.dataDir, conf.entropyDir, all_scores(i).name));
	    cur_scores = cur_scores.hist;
	    for j=1:num_models
	      to_be_selected = range_ + j;
	      selected = cur_scores(to_be_selected);
        fin_selected(j, cur_id) = selected(run);
	    end 
	  end
	  save(sprintf('%s/%s/%s/%d.mat', conf.dataDir, conf.entropyDir, conf.scoresDir, run),'fin_selected');
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%STEP 2: Write in one file with groundtruth

  num = [];
  for i=1:conf.numClasses
    temp = load(sprintf('%s/%s/combined-models/%d.mat', conf.dataDir, model.path, i)); 
    num = [num size(temp.combined_w, 2)];
  end

  num = cumsum(num);
  num = [0 num];
  
  valSet = find(imdb.images.isTrain==1 & imdb.images.isTrainVal==0);
  gt = imdb.images.class(valSet);
  gt = repmat(gt, [1 5]);
  
  tot_models = [];
  for i=1:5
    a = load(sprintf('%s/%s/%s/%d.mat', conf.dataDir, conf.entropyDir, conf.scoresDir, i));
    tot_models = [tot_models a.fin_selected];
  end

  to_be_rejected = find(tot_models(1,:)==-999); 
  gt(to_be_rejected) = [];
  tot_models(:,to_be_rejected) = [];

  system(sprintf('mkdir -p %s/%s/%s', conf.dataDir, conf.entropyDir, conf.finscoresDir));
  for i=1:conf.numClasses
    fin_scores = tot_models(num(i)+1:num(i+1), : );
    save(sprintf('%s/%s/%s/%d-scores.mat', conf.dataDir, conf.entropyDir, conf.finscoresDir, i),'gt','fin_scores','-v7.3');
  end
end
