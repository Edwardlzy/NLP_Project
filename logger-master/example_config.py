if FLAGS.batch_id == 0:
  ##final search 0
  BATCH_ID = 0
  BINARY = "example_cifar_dist_sgd_avg.py"
  LOCAL_GPU_ID = [0, ]
  EXP_NAME = "cifar10_noisy"
  EXTRA_PARAM = ["--useDataProvider", "--maxiter 100000", "--num_gpus 1"]

  SEARCH_PARAMS = {
  '--batchSize':[256,],
  '--eps': [1e-2,],
  '--clipKL': [10.,],
  'model': [
              "--useResNet --resnet_size 32", 
           ],
  }
  


