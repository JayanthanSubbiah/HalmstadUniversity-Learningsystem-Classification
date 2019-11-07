
mdl = fitlm (wineTrainX, wineTrainY);
plot(mdl)
CM = mdl.CoefficientCovariance;
mdk = fitcknn(wineTrainX, wineTrainY);
cvmdl = crossval(mdk);
cvmdlloss = kfoldLoss(cvmdl);
