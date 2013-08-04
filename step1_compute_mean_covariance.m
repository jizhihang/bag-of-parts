function step1_compute_mean_covariance 
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  stacked_hog = {};

  % Sample windows at each scale from all the Training Images
  % Number of windows at scale s = 50/s

  trainSet = find(imdb.images.isTrainVal);
  num_windows = round(50 ./ model.scales);

  for i=1:length(trainSet)
    im = imread(fullfile(conf.imgDir, imdb.images.name{trainSet(i)}));

    temp_stacked_hog = [];
    for s = 1:numel(model.scales)
      
      ims = imresize(im2single(im), 1/model.scales(s)) ;
      hog = vl_hog(ims, model.cellSize) ;
      
      start_x = 1;
      start_y = 1;
      end_x   = size(hog,2) - model.width  + 1;
      end_y   = size(hog,1) - model.height + 1;
      
      blocks_x = [start_x : end_x];
      blocks_y = [start_y : end_y];

      [p,q] = meshgrid(blocks_y, blocks_x);

      pairs = [p(:) q(:)];

      random_blocks = randperm(size(pairs,1));
      random_blocks = random_blocks(1 : min(length(random_blocks), num_windows(s)));
     
      hog_blocks = zeros(length(random_blocks), model.length, 'single'); 

      for jj = 1 : length(random_blocks)
        cur_x = pairs(random_blocks(jj),2);
        cur_y = pairs(random_blocks(jj),1);
        temp_hog = hog(cur_y : cur_y + model.height - 1, cur_x : cur_x + model.width - 1, :);
        temp_hog = reshape(temp_hog, [1 model.length]);
        hog_blocks(jj,:) = temp_hog;
      end
      temp_stacked_hog = [temp_stacked_hog ; hog_blocks];

    end
    stacked_hog{i} = temp_stacked_hog;
  end

  stacked_hog = cat(1, stacked_hog{:});
  covariance = cov(stacked_hog);
  mean_ = mean(stacked_hog);

  save(fullfile(conf.dataDir, 'mean.mat'),'mean_');
  save(fullfile(conf.dataDir, 'covariance.mat'),'covariance');
  save(fullfile(conf.dataDir, 'stacked_hog.mat'),'stacked_hog','-v7.3');

end 
