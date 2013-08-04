function step3_learn_classifiers_fast(class)
  load_vlfeat('0.9.16');
  config;

  system(sprintf('mkdir -p %s/%s/%d', conf.dataDir, model.path, class));

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  load(fullfile(conf.dataDir, 'mean.mat'));
  load(fullfile(conf.dataDir, 'covariance.mat'));
  covariance = covariance + 0.01 * (eye(size(covariance)));

  model.w         = randn(model.width*model.height*31 ,1) ;

  seed_blocks = load(fullfile(conf.dataDir, conf.superpixelsDir, 'selected-blocks', sprintf('%d.mat',class)));
  seed_blocks = seed_blocks.tot_selected_blocks;
  
  selPos = find(imdb.images.isTrainVal & imdb.images.class == class) ;
  selNegT= find(imdb.images.isTrainVal & imdb.images.class ~= class) ;
 
  fprintf('class=%d\n',class);

  pos_ = selPos ;
  [pos_, posDescrs_] = getDescriptors2(conf, imdb, model, pos_) ;
  pos_ = single(pos_); 
  for i= 1 : size(seed_blocks,1)
    fprintf('Block-id:%d\n',i);
    if(exist(sprintf('%s/%s/%d/%d.mat', conf.dataDir, model.path, class, i)))
      continue
    end
    pos = seed_blocks(i, :)';
    [pos, posDescrs] = getDescriptors(conf, imdb, model, pos) ;
    mu_1 = posDescrs;
    mu_0 = mean_';
    model.w = covariance\( mu_1 - mu_0);

    for lt=1:10
	    scores_ = model.w'*posDescrs_ ;
	    new = ~ismember(pos_', pos', 'rows') ;
	    scores_(~new) = -inf ;
	    [drop, perm] = sort(scores_, 'descend') ;
	    pos = cat(2,pos, pos_(:, perm(1:10))) ;
	    posDescrs = cat(2,posDescrs, posDescrs_(:,perm(1:10))) ;
      mu_1 = mean(posDescrs, 2);
      model.w = covariance\( mu_1 - mu_0);
    end
    save(sprintf('%s/%s/%d/%d.mat', conf.dataDir, model.path, class, i), 'model');
  end
end

% --------------------------------------------------------------------
function [frames, descrs] = getDescriptors(conf, imdb, model, frames)
% --------------------------------------------------------------------
  fillFrames = (size(frames,1) == 1) ;
  descrs = zeros(numel(model.w), size(frames,2), 'single') ;
  
  id = 0 ;
  for k = 1:size(frames,2)
    if frames(1,k) ~= id
      id = frames(1,k) ;
      ii = vl_binsearch(imdb.images.id, id);
      im = imread(fullfile(imdb.dir, imdb.images.name{ii})) ;
  
      for s = 1:numel(model.scales)
        ims = imresize(im2single(im), 1/model.scales(s)) ;
        hog{s} = vl_hog(ims, model.cellSize) ;
        %%%%%%%%%%% Discard low strength magnitudes blocks
          if(size(ims,3)==1)
              temp_im = ims;
          else
              temp_im = rgb2gray(ims);
          end
          hx = [-1,0,1];
          hy = -hx';
          grad_xr = imfilter(double(temp_im),hx);
          grad_yu = imfilter(double(temp_im),hy);
          grad_xr = grad_xr(2:end-1, 2:end-1);
          grad_yu = grad_yu(2:end-1, 2:end-1);
          magnit =((grad_yu.^2)+(grad_xr.^2)).^.5;
          magnit = single(padarray(magnit, [1 1]));
          grad_w = single(ones(8,8));
          magnit = vl_fconv(magnit,grad_w);
          magnit = magnit(1:8:end,1:8:end);
          if(size(magnit,1) < 8 || size(magnit,2)<8)
              continue
          end
          magnit = vl_fconv(magnit,grad_w);
          grad{s}=ones(size(magnit));
          grad{s}(find(magnit<=50)) = 0;
        %%%%%%%%%%% Discard low strength ends
      end
	    if fillFrames
	      w = model.w;
	      w = single(reshape(w,model.height,model.width,[])) ;
	      sel0 = find(frames(1,:) == id) ;
	      n = numel(sel0) ;
	      f = zeros(4,0) ;
	      for s = 1:numel(hog)
	        if(size(hog{s},1) < 8 || size(hog{s},2)<8)
	            continue
	        end
	        scores = vl_fconv(hog{s}, w) ;
	        if(size(scores,1) + size(scores,2) < n)
	            continue;
	        end
	        [~,perm] = sort(scores(:)', 'descend') ;
	        %%%%%%%%%%% Discard low strength magnitudes blocks
	        tmp_idx = find(grad{s} == 0);
	        tmp_idx = find(ismember(perm,tmp_idx));
	        perm(tmp_idx) = [];
	        %%%%%%%%%%% Discard low strength ends  
	
	        [perm,sel] = vl_colsubset(perm, n, 'beginning') ;
	        f(1,end+1:end+n) = id ;
	        [f(3,end-n+1:end), ...
	         f(2,end-n+1:end)] = ind2sub([size(scores,1) size(scores,2)], perm) ;
	        f(4,end-n+1:end) = s ;
	        f(5,end-n+1:end) = scores(sel) ;
	      end
	      [~,perm] = sort(f(5,:),'descend') ;
	      frames(1:4,sel0) = f(1:4,perm(1:n)) ;
	      frames(2:3,sel0) = frames(2:3,sel0) - 1 ;
	    end
    end
    tmp = hog{frames(4,k)}(frames(3,k) + (1:model.height), ...
                         frames(2,k) + (1:model.width), ...
                         :) ;
    descrs(1:end, k) = tmp(:) ;
  end
end

% --------------------------------------------------------------------
function [frames, descrs] = getDescriptors2(conf, imdb, model, input)
% --------------------------------------------------------------------
descrs  = [];
frames = [];
  for k = 1:length(input)
    k
    id = input(1,k) ;
    ii = vl_binsearch(imdb.images.id, id);
    im = imread(fullfile(imdb.dir, imdb.images.name{ii})) ;
    for s = 1:numel(model.scales)
      s
      ims = imresize(im2single(im), 1/model.scales(s)) ;
      hog{s} = vl_hog(ims, model.cellSize) ;
      %%%%%%%%%%% Discard low strength magnitudes blocks
      if(size(ims,3)==1)
          temp_im = ims;
      else
          temp_im = rgb2gray(ims);
      end
      hx = [-1,0,1];
      hy = -hx';
      grad_xr = imfilter(double(temp_im),hx);
      grad_yu = imfilter(double(temp_im),hy);
      grad_xr = grad_xr(2:end-1, 2:end-1);
      grad_yu = grad_yu(2:end-1, 2:end-1);
      magnit = ((grad_yu.^2)+(grad_xr.^2)).^.5;
      magnit = single(padarray(magnit, [1 1]));
      grad_w = single(ones(8,8));
      magnit = vl_fconv(magnit,grad_w);
      magnit = magnit(1:8:end,1:8:end);
      if(size(magnit,1) < 8 || size(magnit,2)<8)
          continue
      end
      magnit = vl_fconv(magnit,grad_w);
      grad{s}=ones(size(magnit));
      grad{s}(find(magnit<=50)) = 0;
      perm = [1:(size(hog{s},1)-7)*(size(hog{s},2)-7)];
      tmp_idx = find(grad{s} == 0);
      tmp_idx = find(ismember(perm,tmp_idx))      ;
      perm(tmp_idx) = [];
      %%%%%%%%%%% Discard low strength ends  
      n = length(perm);
      temp_descrs = single(zeros(model.length, n));
      temp_frames = uint32(zeros(4, n));
      for jjj=1:n
        fprintf('%d/%d %d %d/%d\n',k, length(input),s, jjj,n);
        [x, ...
         y] = ind2sub([size(hog{s},1)-7 size(hog{s},2)-7], perm(jjj)) ;

        x = x -1;
        y = y -1;
        temp_frames(:,jjj) =  [id ;y; x; s];
        tmp = hog{s}(x + (1:model.height), ...
                          y + (1:model.width), ...
                        :) ;
        temp_descrs(:,jjj) = tmp(:);
      end
      descrs = [descrs temp_descrs];
      frames = [frames temp_frames];
    end
  end
end

