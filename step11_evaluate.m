function step11_evaluate(num_classifiers)
	load_vlfeat('0.9.14');
	config;
	
	imdb = load(fullfile(conf.dataDir, conf.imdb));
	
	categories = imdb.classes;
	
	fid = fopen(sprintf('%s', conf.testImageFile), 'r');
	[testSet] = textscan(fid,'%s','Delimiter','\n');
	testSet = testSet{1};
	fclose(fid);
	
	num_categories = length(categories);
	
	tot_scores = [];
	tot_prob = [];
	APval = 0;
	for i = 1:length(categories)
	  fid = fopen(sprintf('%s/%s/results/%s-results_num-classifiers-%d.txt', conf.dataDir, conf.bopDir, categories{i}, num_classifiers));
	  lines = textscan(fid, '%s', 'Headerlines', 1, 'Delimiter','\n');
	  lines = lines{1};
	  frames = {};
  	prob=[];
  	gt = [];
  	scores = [];
		for jj=1:length(lines)
  	  temp_line = regexp(lines{jj},',','split');
  	  frames{jj} = temp_line{1};
  		scores(jj) = str2double(temp_line{2});
  		gt(jj) = str2double(temp_line{3});
  		prob(jj) = str2double(temp_line{4});
  	end
		scores = scores';
  	gt = gt';
  	prob = prob';
  	[ok, where] = ismember( testSet, frames);
  	scores =scores(where);
  	prob =prob(where);
  	tot_scores = [tot_scores scores];
		tot_prob = [tot_prob prob];
  	[tempcat tempAP] = textread(sprintf('%s/%s/results/%s-results_num-classifiers-%d.txt', conf.dataDir, conf.bopDir,categories{i}, num_classifiers), '%s %s',1);
	
	  [garbage tempAP] = strtok(tempAP,'=');
		[tempAP garbage] = strtok(tempAP,'=');
  	APval = APval + str2double(tempAP);
  	fclose(fid);
  end

	[max_scores, computed_labels]=max(tot_scores,[],2);
	[max_prob, computed_labels]=max(tot_prob,[],2);
	groundTruth = zeros(length(testSet),1);
	
	for i =1:length(testSet)
	  parts = regexp(testSet{i},'/','split');
	  category = parts{1};
	  [ok, where] = ismember (category , categories);
  	groundTruth(i) = where;
	end
	
	confusion_matrix = zeros(num_categories, num_categories);
	for j = 1:num_categories
		for i = 1:num_categories
	  	confusion_matrix(i,j) =  ( sum((groundTruth==i).*(computed_labels==j)) / (sum(groundTruth==i)));
  	end
	end
	cur_AP = APval/num_categories;
	cur_accuracy = 100*mean(diag(confusion_matrix));
	fprintf('Num-parts=%d Accuracy=%f AP=%f\n', num_classifiers, cur_accuracy, cur_AP);
end
