conf.randSeed = 1 ;
conf.dataDir  = 'data/';
conf.imgDir   = 'data/scene67/Images';
conf.imdb     = 'scene67-imdb.mat';
conf.superpixelsDir = 'superpixels';
conf.scales = logspace(log10(1), log10(2), 4) ;
conf.segmentPath = './helper/segment'; % Path to superpixel binary
conf.numClasses = 67 ;
conf.entropyDir = 'entropy';
conf.scoresDir = 'scores';
conf.finscoresDir = 'fin_scores';
conf.numSpatialX = [1 2 ];
conf.numSpatialY = [1 2 ];
conf.trainImageFile = 'data/scene67/TrainImages.txt';
conf.testImageFile  = 'data/scene67/TestImages.txt';
conf.bopDir = 'bag-of-parts';

% model parameter
model.path = 'multiscale-blocks/models' ;
model.cellSize = 8 ;
model.scales   = logspace(log10(1), log10(2), 4) ;
model.width    = 8 ;
model.height   = 8 ;
model.dims     = 31;
model.length   = model.width * model.height * model.dims;
model.biasMultiplier = 100 ;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

addpath helper;
