step1_compute_mean_covariance;

for class=1:67
  step2_compute_superpixels(class, 0.5, 500, 20);
end

for class=1:67
  step3_learn_classifiers_fast(class);
end

step4_0_compute_hog;

for class=1:67
  step4_compute_validation_scores(class);
end

step5_0_combine_models;

for img_idx=1:1072
  step5_1_compute_entropy(img_idx);
end

step5_2_compute_entropy_combine;

for class=1:67
  step5_3_compute_entropy_final(class);
end

step6_select_classifiers(100)

for class=1:67
  step7_compute_bag_of_parts(100)
end

step7_1_compute_jittered_bop(class, 100)

for class=1:67
  for C = [ 1 , 10, 100, 1000, 10000]
    for num_classifiers = [10 : 10: 100]
      step8_cross_validation(class, C, num_classifiers)  
    end
  end
end

step9_find_best_c;

step10_best_c_jobs;

for num_classifiers = [10 : 10: 100]
  step11_evaluate(num_classifiers);
end
