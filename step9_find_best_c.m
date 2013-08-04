function step9_find_best_c 
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));
  categories = imdb.classes;

	num_classifiers_range = [10:10:100];
	C_range = [ 1 , 10, 100, 1000, 10000];
	
	fid = fopen('step10_best_c_jobs.m','w');
	count = 1;
	
	for num_classifiers = num_classifiers_range
	  for cat=1:length(categories)
	    cur_cat = categories{cat};
	    AP = [];
	    fB = [];
	    fA = [];
	    for C = C_range
	      load(sprintf('%s/%s/cross-validation/%s-%f-%d-sigmoid-AP.mat',conf.dataDir, conf.bopDir, cur_cat,C,num_classifiers));
	      AP = [AP mean(finAP)];
	      fA  = [fA mean(finA)];
	      fB  = [fB mean(finB)];
	    end
	    [maxAP, idx] = max(AP);
	    fA = fA(idx);
	    fB = fB(idx);
	    bestC = C_range(idx);

	    fprintf(fid,'train_test(%d, %d, %f, %f, %f);\n',cat,num_classifiers,bestC,fA,fB);
	    count = count+1;
	  end
	end
	fclose(fid);
end
