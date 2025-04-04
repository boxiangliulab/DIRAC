def supervised_hyperparams(lr=1e-3, wd=5e-5, scheduler=True):
    """
    Return a dictionary of hyperparameters for the Supervised algorithm.
    Default parameters are the best ones as found through a hyperparameter search.
    Arguments:
    ----------
    lr: float
        Learning rate.
    wd: float
        Weight decay for the optimizer.
    scheduler: bool
        Will use a OneCycleLR learning rate scheduler if set to True.
    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on Supervised.
    """

    hyperparams = {'learning_rate': lr,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler
                   }

    return hyperparams

def adda_hyperparams(lr_target=1e-5, lr_discriminator=1e-4, wd=5e-5, scheduler=False):
    """
    Return a dictionary of hyperparameters for the ADDA algorithm.
    Default parameters are the best ones as found through a hyperparameter search.
    Arguments:
    ----------
    lr_target: float
        Learning rate for the target encoder.
    lr_discriminator: float
        Learning rate for the discriminator.
    wd: float
        Weight decay for the optimizer.
    scheduler: bool
        Will use a OneCycleLR learning rate scheduler if set to True.
    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on ADDA.
    """
    
    hyperparams = {'learning_rate_target': lr_target,
                   'learning_rate_discriminator': lr_discriminator,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler
                   }

    return hyperparams

def unsuper_hyperparams(lr=1e-3, tau=0.9, wd=5e-3, scheduler=True):
    """
    Return a dictionary of hyperparameters for the AdaMatch algorithm.
    Default parameters are the best ones as found through a hyperparameter search.
    Arguments:
    ----------
    lr: float
        Learning rate.
    tau: float
        Weight of the unsupervised loss.
    wd: float
        Weight decay for the optimizer.
    scheduler: bool
        Will use a OneCycleLR learning rate scheduler if set to True.
    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on AdaMatch.
    """
    
    hyperparams = {'learning_rate': lr,
                   'tau': tau,
                   'weight_decay': wd,
                   'step_scheduler': scheduler
                   }

    return hyperparams

def self_ensemble_hyperparams(lr=1e-3, unsupervised_weight=3.0, wd=5e-5, scheduler=False):
    """
    Return a dictionary of hyperparameters for the Self-Ensemble algorithm.
    Default parameters are the best ones as found through a hyperparameter search.
    Arguments:
    ----------
    lr: float
        Learning rate.
    unsupervised_weight: float
        Weight of the unsupervised loss.
    wd: float
        Weight decay for the optimizer.
    scheduler: bool
        Will use a OneCycleLR learning rate scheduler if set to True.
    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on ADDA.
    """
    
    hyperparams = {'learning_rate': lr,
                   'unsupervised_weight': unsupervised_weight,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler
                   }

    return hyperparams


def cgdm_hyperparams(lr=1e-3, wd=5e-5, num_k=3, scheduler=False, pseudo_interval=5):

    hyperparams = {'learning_rate': lr,
                   'num_k': num_k,
                   'weight_decay': wd,
                   'cyclic_scheduler': scheduler,
                   'pseudo_interval': pseudo_interval,
                   }
    return hyperparams