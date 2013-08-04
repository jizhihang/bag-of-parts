function step8_cross_validation(category_idx, C, num_classifiers)
  load_vlfeat('0.9.14');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  category = imdb.classes{category_idx};

  if(exist(sprintf('%s/%s/cross-validation/%s-%f-%d-sigmoid-AP.mat',conf.dataDir, conf.bopDir, category,C,num_classifiers)))
    return
  end
  system(sprintf('mkdir -p %s/%s/cross-validation',conf.dataDir, conf.bopDir));

	finA = [];
	finB = [];
	finAP = [];

  range_ = [1:100:33500];

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

  clear bag_of_words bag_of_parts 

  jittered_bop = load(sprintf('%s/%s/jittered-features/%s-hists.mat', conf.dataDir, conf.bopDir, category) );
  j_features = single(jittered_bop.hists);
  features_ = [];
  for i =1:length(range_)
    features_ = [features_; j_features(range_(i):range_(i)+num_classifiers-1,:)];
  end
  j_features = features_;

  p = 1;
  pNorm = sum(abs(j_features).^p,1).^(1/p);
  j_features = bsxfun(@rdivide,j_features, pNorm);
  jittered_bop= j_features;

  jittered_features =  jittered_bop;

  fid = fopen(sprintf('%s', conf.trainImageFile), 'r');
	[gtrainList] = textscan(fid,'%s','Delimiter','\n');
	gtrainList = gtrainList{1};
	fclose(fid);
	
	gposTrainList = regexp(gtrainList, sprintf('^%s/', category));
	gnegTrainList = cellfun('isempty',gposTrainList);
	gposTrainList = find(~gnegTrainList);
	gnegTrainList = find(gnegTrainList);
	
	randPosTrain = randperm(length(gposTrainList));
	randNegTrain = randperm(length(gnegTrainList));

  for my_run=1:4
    pos_start = (my_run-1)*20 + 1;
    pos_end   = min((my_run)*20, length(gposTrainList));

    neg_start = (my_run-1)*1320 +1;
    neg_end   = min((my_run)*1320, length(gnegTrainList));

    pos_range = [pos_start : pos_end];
    neg_range = [neg_start : neg_end];
    tot_pos_range = 1:length(gposTrainList);
    tot_neg_range = 1:length(gnegTrainList);

    sPosTrainList = gposTrainList(randPosTrain(setdiff(tot_pos_range, pos_range)));
    sPosTestList  = gposTrainList(randPosTrain(pos_range));
    sNegTrainList = gnegTrainList(randNegTrain(setdiff(tot_neg_range, neg_range)));
    sNegTestList  = gnegTrainList(randNegTrain(neg_range));

    trainList = [sPosTrainList ; sNegTrainList];
    testList  = [sPosTestList ; sNegTestList];
    train_features = features(:,trainList);
    test_features = features(:,testList);

    posTrainList = [1:length(sPosTrainList)];
    negTrainList = [length(sPosTrainList)+1:length(trainList)];

    posTestList = [1:length(sPosTestList)];
    negTestList = [length(sPosTestList)+1:length(testList)];

    trainLabels = int8(ones(length(trainList),1));
    testLabels  = int8(ones(length(testList),1));
    trainLabels(negTrainList) = -1;
    testLabels(negTestList) = -1;

    s_jittered_features = jittered_features(:,randPosTrain(setdiff(tot_pos_range, pos_range)));
    train_features = [train_features s_jittered_features];
    trainLabels = [trainLabels ;ones(length(posTrainList),1)];

    psix = vl_homkermap(train_features, 1, 'Kchi2') ;
    conf.svm.C = C;
    conf.num_iter = 100;
    numPos = 2*length(posTrainList);
    num= length(trainList) + length(posTrainList);

    conf.svm.lambda = 1 / (conf.svm.C *  num) ;
    lambda = conf.svm.lambda;
    conf.svm.bias = 1;
    dimension = size(psix,1);
    w = zeros(dimension+1,1,'single') ;
    prec = ones(dimension+1,1,'single') ; prec(end) = .1/conf.svm.bias ;
    
    weight_vectors = zeros(dimension+1,conf.num_iter,'single') ;
    num = length(trainList) + length(posTrainList);
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

    psix = vl_homkermap(test_features, 1, 'Kchi2') ;
    models = mean(weight_vectors(:,end-19:end),2);

    scores = models(1:end-1,1)' * psix + models(end,1) * conf.svm.bias ;
    [r, p, info] = vl_pr(testLabels,scores);
    AP = info.auc*100;

    out = scores;
    target = testLabels;
    prior1 = length(find(target==1));
    prior0 = length(find(target==-1));

    [A,B] = fit_sigmoid(out, target);
    finA = [finA A];
    finB = [finB B];
    finAP= [finAP AP];
  end
  A = mean(finA); B= mean(finB);
  save(sprintf('%s/%s/cross-validation/%s-%f-%d-sigmoid-AP.mat',conf.dataDir, conf.bopDir, category,C,num_classifiers), 'A', 'B', 'finA','finB','finAP');
end
