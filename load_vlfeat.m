function load_vlfeat(version)
  addpath(sprintf('vlfeat-%s/toolbox/', version));
  vl_setup;
end
