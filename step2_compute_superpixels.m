function step2_compute_superpixels(class, sig, k, mint)
  load_vlfeat('0.9.16');
  config;

  imdb = load(fullfile(conf.dataDir, conf.imdb));
  
  trainList = find(imdb.images.isTrainVal==1 & imdb.images.class==class);

  system(sprintf('mkdir -p %s', fullfile(conf.dataDir, conf.superpixelsDir, 'temp')));
  system(sprintf('mkdir -p %s', fullfile(conf.dataDir, conf.superpixelsDir, 'selected-blocks')));

  tot_selected_blocks = [];

  for iii=1:length(trainList)
      cur = trainList(iii);
      im = imread(fullfile(conf.imgDir, imdb.images.name{cur}));
      tempname = fullfile(conf.dataDir, conf.superpixelsDir, 'temp', sprintf('%d',class));

      for scale_idx  = 1:length(conf.scales)
		    img = imresize(im,1/conf.scales(scale_idx));
  			inpath  = [tempname 'in.ppm'];
  			outpath = [tempname 'out.ppm'];
				imwrite(img,inpath);
		    cmd_str = sprintf('%s %f %f %f %s %s', conf.segmentPath, sig, k, mint, inpath, outpath);
  			system(cmd_str);
  			seg_img = imread(outpath);
  			delete(inpath);
  			delete(outpath);
  			rgb = reshape(seg_img,[],3);
  			[u, m, n] = unique(rgb,'rows');
  			sup_seg = reshape(n,size(img,1),size(img,2));
  	    selected_x = []; selected_y= [];
		    unique_sup = unique(sup_seg);
		    for i=1:length(unique_sup)
	        idx = find(sup_seg == unique_sup(i));          
	        [all_y all_x] = ind2sub([size(img,1) size(img,2)], idx);
	        center_y = round(mean(all_y));
	        center_x = round(mean(all_x));
	        cur_area = length(idx);
	        flag = 0;
	        if(cur_area < 1500 && cur_area > 500) 
	          selected_x = [selected_x center_x];
	          selected_y = [selected_y center_y];
	        end
	    end

	    for i=1:length(selected_y)
        cropped = imcrop(img, [selected_x(i)-32 selected_y(i)-32 63 63]);
        if(size(cropped,1)~=64 || size(cropped,2)~=64)
          continue 
        end
        block_idx_x = ceil((selected_x(i) - 32)/8) - 1;
        block_idx_y = ceil((selected_y(i) - 32)/8) - 1;
        if(filter_low_grad(cropped)==0)
          continue
        end
        tot_selected_blocks = [tot_selected_blocks; [cur block_idx_x block_idx_y scale_idx]];
	    end
	
    end
  end
	save(fullfile(conf.dataDir, conf.superpixelsDir, 'selected-blocks', sprintf('%d.mat', class)), 'tot_selected_blocks');
end

function ret_val = filter_low_grad(im)
	im = im2single(im);
	if(size(im,3)~=1)
	    temp_im = rgb2gray(im);
	else
	    temp_im = im;
	end
	hx = [-1,0,1];
	hy = -hx';
	grad_xr = imfilter(double(temp_im),hx);
	grad_yu = imfilter(double(temp_im),hy);
	grad_xr = grad_xr(2:end-1, 2:end-1);
	grad_yu = grad_yu(2:end-1, 2:end-1);
	magnit =((grad_yu.^2)+(grad_xr.^2)).^.5;
	if(sum(magnit(:)) < 50)
	  ret_val= 0 ;
	else
	  ret_val = 1;
	end
end

function [coordX, coordY] = square(x_c, y_c, A)
	% Generates the coordinates of a square with center (x_c, y_c) and side A.
	
	startx  = x_c - A/2;
	endx    = x_c + A/2;
	starty  = y_c - A/2;
	endy    = y_c + A/2;
	
	% Left side
	tempx1 = [startx startx];
	tempy1 = [starty endy];
	
	% Top side
	tempx2 = [startx endx];
	tempy2 = [endy endy];
	
	% Right side
	tempx3 = [endx endx];
	tempy3 = [endy starty];
	
	% Bottom side
	tempx4 = [endx startx];
	tempy4 = [starty starty];
	
	coordX = [tempx1 tempx2 tempx3 tempx4];
	coordY = [tempy1 tempy2 tempy3 tempy4];
end

