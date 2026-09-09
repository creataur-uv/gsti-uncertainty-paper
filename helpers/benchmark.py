
def benchmark(obs, pred):
    """
        code to test model performance based on 
        RMSE:   Root Mean Squared Error
        MAE:    Mean Absolute Error
        R2:     r2_score
        EVS:    Explained Variance Score


        input:
        -----
            obs:    observed values
            pred:   values predicted using model
    """
    from sklearn import metrics
    import numpy

    res: dict[str, float] = {}
    res["RMSE"] = metrics.root_mean_squared_error(obs, pred)
    mean_obs = sum(obs)/len(obs)
    res["nRMSE_avg"] = res["RMSE"]/ mean_obs
    res["R2"] = metrics.r2_score(obs, pred)
    res["MAE"] = metrics.mean_absolute_error(obs, pred)
    res["EVS"] = metrics.explained_variance_score(obs, pred)

    return res

def benchmark_new(obs, pred, sigma):
    """
        code to test model performance based on 
        RMSE:   Root Mean Squared Error
        MAE:    Mean Absolute Error
        R2:     r2_score
        EVS:    Explained Variance Score
        nRMSE:  RMSE/mean(observed_values)
        PICP:   Prediction Interval Coverage Probability
        MPIW:   Mean Prediction Interval Width


        input:
        -----
            obs:    observed values
            pred:   values predicted using model
    """
    from sklearn import metrics
    import numpy

    res: dict[str, float] = {}
    res["RMSE"] = metrics.root_mean_squared_error(obs, pred)
    res["nRMSE_avg"] = res["RMSE"]/ numpy.mean(obs)
    res["nRMSE_std"] = res["RMSE"]/ numpy.std(obs)
    res["nRMSE_range"] = res["RMSE"]/ numpy.ptp(obs)
    res["R2"] = metrics.r2_score(obs, pred)
    res["MAE"] = metrics.mean_absolute_error(obs, pred)
    res["EVS"] = metrics.explained_variance_score(obs, pred)
    res["PICP"], res["MPIW"] = uncertainty_quantifier(obs, pred, sigma)

    return res

def uncertainty_quantifier(true, pred, pred_std):
    if (len(true) != len(pred)) or (len(true) != len(pred_std)):
        return 

    range_preds = list()
    coverage_proportion = list()

    for i in range(len(true)):
        U_i = pred[i] + 1.96*pred_std[i]    
        L_i = pred[i] - 1.96*pred_std[i]
        range_preds.append(U_i - L_i)
        if L_i <= true[i] <= U_i:
            coverage_proportion.append(1)
        else:
            coverage_proportion.append(0)

    PICP = sum(coverage_proportion) / len(true)
    MPIW = sum(range_preds) / len(true)

    return PICP, MPIW