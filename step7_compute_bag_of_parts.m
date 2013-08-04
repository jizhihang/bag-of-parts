function step7_compute_bag_of_parts
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  num_classifiers = 100;

  system(sprintf('mkdir -p %s/%s/features', conf.dataDir, conf.bopDir));
  load(sprintf('%s/%s/selected_classifiers.mat', conf.dataDir, conf.entropyDir));
	
  combined_models = [];
	for i=1:conf.numClasses
	    for j=1:num_classifiers
	        fprintf('%d %d\n',i,selected_classifiers{i}(j));
	        a = load(sprintf('%s/%s/%d/%d.mat', conf.dataDir, model.path, i, selected_classifiers{i}(j)));
	        combined_models = [combined_models a.model.w];
	    end
	end

  fid = fopen(sprintf('%s', conf.trainImageFile), 'r');
  [trainList] = textscan(fid,'%s','Delimiter','\n');
  trainList = trainList{1};
  fclose(fid);
  fid = fopen(sprintf('%s', conf.testImageFile), 'r');
  [testList] = textscan(fid,'%s','Delimiter','\n');
  testList = testList{1};
  fclose(fid);

  model.w = [];

  hist = {};
	for i=1:length(trainList)
	    fprintf('%d/%d\n',i,length(trainList));
	  	ii = trainList{i};
  		im = imread(fullfile(imdb.dir, ii)) ;
      [frames, descrs, hog_size, img_size] = getDescriptors(im, model, conf);
      combined_scores = descrs' * combined_models;
      hist{i} = getHist(combined_scores, num_classifiers, conf, frames, hog_size);
	end
  hists = cat(2,hist{:});
  save(sprintf('%s/%s/features/train_hists.mat', conf.dataDir, conf.bopDir),'hists','-v7.3');
 

  hist = {};
	for i=1:length(testList)
	  fprintf('%d/%d\n',i,length(testList));
		ii = testList{i};
		im = imread(fullfile(imdb.dir, ii)) ;
      [frames, descrs, hog_size, img_size] = getDescriptors(im, model, conf);
      combined_scores = descrs' * combined_models;
      hist{i} = getHist(combined_scores, num_classifiers, conf, frames, hog_size);
	end
  hists = cat(2,hist{:});
  save(sprintf('%s/%s/features/test_hists.mat', conf.dataDir, conf.bopDir),'hists','-v7.3');
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
  
          x= x -1;
          y = y -1;
          tmp = hog{s}(x + (1:model.height), ...
                            y + (1:model.width), ...
                          :) ;
          temp_descrs(:,jjj) = tmp(:);
          temp_frames(:,jjj) = [x ;y ;s];
        end
        descrs = [descrs temp_descrs];
        frames = [frames temp_frames];

	    end
      img_size = size(im);
end


function hist = getHist(combined_scores, num_classifiers, conf, frames, hog_size)
          for i = 1:length(conf.numSpatialX)
            hist = zeros(conf.numSpatialY(i) * conf.numSpatialX(i) , conf.numClasses * num_classifiers) ;
            for jj=1:conf.numSpatialY(i) * conf.numSpatialX(i)
              temp_scores = [];
              aaa = [];
              for s=1:4
                cur_frames = frames(:,find(frames(3,:) == s));
                cur_scores = combined_scores(find(frames(3,:) == s),:);
                binsx = vl_binsearch(linspace(0,hog_size{s}(2),conf.numSpatialX(i)+1), cur_frames(2,:)) ;
                binsy = vl_binsearch(linspace(0,hog_size{s}(1),conf.numSpatialY(i)+1), cur_frames(1,:)) ;
                % combined quantization
                bins = sub2ind([conf.numSpatialY(i), conf.numSpatialX(i)], ...
                 binsy,binsx) ;
                temp_scores = [temp_scores ; cur_scores(find(bins==jj),:)];
                aaa = [aaa find(bins==jj)];
              end
              hist(jj,:) = max(temp_scores)';
            end
            hists{i} = hist(:);
          end
          hist = cat(1,hists{:}) ;
end
