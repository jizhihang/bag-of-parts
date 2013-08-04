function train_test(category_idx, num_classifiers, C, A, B)
  load_vlfeat('0.9.14');
  config;

  system(sprintf('mkdir -p %s/%s/results', conf.dataDir, conf.bopDir));

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  category = imdb.classes{category_idx};

  range_ = [1:100:33500];

  % Load train hists
  bag_of_parts = load(sprintf('%s/%s/features/train_hists.mat', conf.dataDir, conf.bopDir));
  features = [ bag_of_parts.hists];
  features = single(features);
  features_ = [];
  for i =1:length(range_)
    features_ = [features_; features(range_(i):range_(i)+num_classifiers-1,:)];
  end
  features = features_;
  p = 1;
  pNorm = sum(abs(features).^p,1).^(1/p);
  features = bsxfun(@rdivide,features, pNorm);
  train_features = features;

  % Load test hists
  bag_of_parts = load(sprintf('%s/%s/features/test_hists.mat', conf.dataDir, conf.bopDir));
  features = [ bag_of_parts.hists];
  features = single(features);
  features_ = [];
  for i =1:length(range_)
    features_ = [features_; features(range_(i):range_(i)+num_classifiers-1,:)];
  end
  features = features_;
  p = 1;
  pNorm = sum(abs(features).^p,1).^(1/p);
  features = bsxfun(@rdivide,features, pNorm);
  test_features = features;

  clear bag_of_parts hists features;
 
  % Load jittered features 
  jittered_features = load(sprintf('%s/%s/jittered-features/%s-hists.mat', conf.dataDir, conf.bopDir, category) );
  features = single(jittered_features.hists);
  features_ = [];
  for i =1:length(range_)
    features_ = [features_; features(range_(i):range_(i)+num_classifiers-1,:)];
  end
  features = features_;
  p = 1;
  pNorm = sum(abs(features).^p,1).^(1/p);
  features = bsxfun(@rdivide,features, pNorm);
  jittered_features = features;

  train_features = [train_features jittered_features];

  fid = fopen(sprintf('%s', conf.trainImageFile), 'r');
	[trainList] = textscan(fid,'%s','Delimiter','\n');
	trainList = trainList{1};
	fclose(fid);
	
  fid = fopen(sprintf('%s', conf.testImageFile), 'r');
	[testList] = textscan(fid,'%s','Delimiter','\n');
	testList = testList{1};
	fclose(fid);
 
  posTrainList = regexp(trainList, sprintf('^%s/', category));
  negTrainList = cellfun('isempty',posTrainList);
  posTrainList = find(~negTrainList);
  negTrainList = find(negTrainList);

  posTestList = regexp(testList, sprintf('^%s/', category));
  negTestList = cellfun('isempty',posTestList);
  posTestList = find(~negTestList);
  negTestList = find(negTestList);

  trainLabels = int8(ones(length(trainList),1));
  testLabels = ones(length(testList),1);
  trainLabels(negTrainList) = -1;
  
  trainLabels = [trainLabels ;ones(length(posTrainList),1)];

  testLabels(negTestList) = -1;
  psix = vl_homkermap(train_features, 1, 'Kchi2') ;
  conf.svm.C = C;
  conf.svm.A = A;
  conf.svm.B = B;
  conf.num_iter = 100;
  numPos = 2*length(posTrainList);
  num= length(trainList) + length(posTrainList);

  conf.svm.lambda = 1 / (conf.svm.C *  num) ;
  lambda = conf.svm.lambda;
  conf.svm.bias = 1;
  dimension = size(psix,1);
  w = zeros(dimension+1,1,'single') ;
  prec = ones(dimension+1,1,'single') ; prec(end) = .1/conf.svm.bias ;
  en = zeros(4,conf.num_iter) ;
  b = zeros(1,conf.num_iter) ;
  weight_vectors = zeros(dimension+1,conf.num_iter) ;
  num= length(trainList) + length(posTrainList);

  for t=1:conf.num_iter
    perm = uint32(randperm(num)) ;
    w = vl_pegasos(single(psix),trainLabels,lambda,  ...
    'startingModel', w, ...
    'numIterations', num, ...
    'permutation', perm, ...
    'startingIteration', (t-1)*num+1, ...
    'biasMultiplier', conf.svm.bias, ...
    'preconditioner', prec           ) ;
     weight_vectors(:,t) = w;
  end

  clear psix train_features;
  models = mean(weight_vectors(:,end-19:end),2);

  psix = vl_homkermap(test_features, 1, 'Kchi2') ;

  scores = models(1:end-1,1)' * psix + models(end,1) * conf.svm.bias ;
  [r, p, info] = vl_pr(testLabels,scores);
  AP = info.auc*100;
  fid=fopen((sprintf('%s/%s/results/%s-results_num-classifiers-%d.txt', conf.dataDir, conf.bopDir, category, num_classifiers)),'w') ;
  fin_scores = scores;
  [drop, perm] = sort(fin_scores, 'descend') ;
    fprintf(fid, '%s AP=%g\n', category , AP) ;
  for i=perm
    prob = 1/(1+exp(A*fin_scores(i)+B));
    fprintf(fid, '%s,%g,%d,%f\n', testList{i}, fin_scores(i),testLabels(i), prob) ;
  end
  fclose(fid) ;
end
