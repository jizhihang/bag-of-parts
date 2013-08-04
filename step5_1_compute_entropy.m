function step5_1_compute_entropy(img_idx)
  load_vlfeat('0.9.16');
  config;

  if(exist(sprintf('%s/%s/top5detections/%d.mat', conf.dataDir, conf.entropyDir, img_idx)))
    return
  end

  system(sprintf('mkdir -p %s/%s/top5detections', conf.dataDir, conf.entropyDir));

  imdb = load(fullfile(conf.dataDir, conf.imdb));
	
  valSet = find(imdb.images.isTrain==1 & imdb.images.isTrainVal==0);
  gt = imdb.images.class(valSet)';

  combined_models = load(sprintf('%s/%s/combined-models.mat', conf.dataDir, model.path));
  combined_w = combined_models.combined_w;

  hist = {};
    i = img_idx;
    ii = imdb.images.name{valSet(i)};
    fprintf('%d/%d %s\n',i,length(valSet),ii);
    im = imread(fullfile(imdb.dir, ii)) ;
    [frames, descrs, hog_size, img_size] = getDescriptors(im, model, conf);

    % Do it in chunks of 500
    total_classifiers = size(combined_w,2);
    num_times = floor(total_classifiers/500);
    for jj=1:num_times
        jj
        combined_scores = descrs' * combined_w(:, (jj-1)*500+ 1 : jj*500);
        num_classifiers = 500;
        hist{jj} = single(getHist(combined_scores, num_classifiers, conf, frames, hog_size));
    end
    combined_scores = descrs' * combined_w(:, (num_times * 500) + 1 :end);
    num_classifiers = total_classifiers - (num_times * 500);
    hist{num_times+1} = single(getHist(combined_scores, num_classifiers, conf, frames, hog_size));
    hist = single(cat(1,hist{:}));
    save(sprintf('%s/%s/top5detections/%d.mat', conf.dataDir, conf.entropyDir, img_idx), 'hist');
end


function [frames, descrs, hog_size, img_size] = getDescriptors(im, model, conf)
      frames = [];
      descrs = [];
      hog = {};
      grad={};
      for s = 1:numel(model.scales)
        ims = imresize(im2single(im), 1/model.scales(s)) ;
        hog{s} = vl_hog(ims, model.cellSize) ;
        hog_size{s} = [size(hog{s},1)-7 size(hog{s},2)-7];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        perm = [1:(size(hog{s},1)-7)*(size(hog{s},2)-7)];
        tmp_idx = find(grad{s} == 0);
        tmp_idx = find(ismember(perm,tmp_idx))      ;
        perm(tmp_idx) = [];
        %%%%%%%%%%% Discard low strength ends  
        n = length(perm);
        temp_descrs = single(zeros(model.length, n));
        temp_frames = zeros(3,n);
        for jjj=1:n
          [x, ...
           y] = ind2sub([size(hog{s},1)-7 size(hog{s},2)-7], perm(jjj)) ;
  
          x = x -1;
          y = y -1;
          tmp = hog{s}(x + (1:model.height), ...
                            y + (1:model.width), ...
                          :) ;
          temp_descrs(:,jjj) = tmp(:);
          temp_frames(:,jjj) = [x ;y ;s];
        end
        descrs = [descrs temp_descrs];
        frames = [frames temp_frames];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      end
      img_size = size(im);
end


function hist = getHist(combined_scores, num_classifiers, conf, frames, hog_size)
          for i = 1:length(conf.numSpatialX)
            hist = zeros(conf.numSpatialY(i) * conf.numSpatialX(i) , num_classifiers) ;
            for jj=1:conf.numSpatialY(i) * conf.numSpatialX(i)
              temp_scores = [];
              for s=1:4
                cur_frames = frames(:,find(frames(3,:) == s));
                cur_scores = combined_scores(find(frames(3,:) == s),:);
                binsx = vl_binsearch(linspace(0,hog_size{s}(2),conf.numSpatialX(i)+1), cur_frames(2,:)) ;
                binsy = vl_binsearch(linspace(0,hog_size{s}(1),conf.numSpatialY(i)+1), cur_frames(1,:)) ;
                % combined quantization
                bins = sub2ind([conf.numSpatialY(i), conf.numSpatialX(i)], ...
                 binsy,binsx) ;
                temp_scores = [temp_scores ; cur_scores(find(bins==jj),:)];
              end
              hist(jj,:) = max(temp_scores)';
            end
            hists{i} = hist(:);
          end
          hist = cat(1,hists{:}) ;
end
