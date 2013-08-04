function step4_0_compute_hog 
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));

  valList = find(imdb.images.isTrain == 1 & imdb.images.isTrainVal == 0);

  tot_hog = {};
  tot_grad = {};
  tot_class = {};

  for i=1:length(valList)
    fprintf('%d/%d\n',i,length(valList));
    cur = valList(i);
    im = imread(fullfile(imdb.dir, imdb.images.name{cur})) ;
    class = imdb.images.class(cur);
    hog = {};
    grad = {};
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
    tot_class{i} = class;
    tot_hog{i}   = hog;
    tot_grad{i}  = grad;
  end
  save(sprintf('%s/hog_validation_set_for_learning_block_classifiers.mat', conf.dataDir), 'tot_hog','tot_class','tot_grad');
end

