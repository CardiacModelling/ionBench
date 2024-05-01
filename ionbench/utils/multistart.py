"""
Contains the multistart function (referenced as ionbench.multistart) which can perform multiple runs of the same optimisation function starting at different initial conditions.
"""
import ionbench.utils.cache


def multistart(opt, bm, initParams, filename, **kwargs):
    """
    Run an optimiser multiple times from different starting location.

    Parameters
    ----------
    opt : function
        A function which takes inputs of a benchmarker bm, a vector of initial parameters x0, and returns a vector of optimised parameters. All ionBench optimisers have a .run() function which satisfies these requirements.
    bm : Benchmarker
        A benchmarker problem to use for the optimisation.
    initParams : list
        A list of initial parameter vectors. The first x0 parameters used will be initParams[0].
    filename : string
        A filename to identify this optimisation run. The benchmarker tracker objects will store their data in '[filename]_run[i].pickle' for i in range(len(initParams)).
    kwargs
        Any additional keyword arguments to pass into the opt() call.

    Returns
    -------
    outs : list
        A list of optimised parameters. The same size as initParams.
    """
    outs = []
    for i in range(len(initParams)):
        out = opt(bm, x0=initParams[i], **kwargs)
        if not filename == '':
            bm.tracker.save(filename + '_run' + str(i) + '.pickle')
        print(out)
        outs.append(out)
        bm.reset(fullReset=False)
        ionbench.utils.cache.clear_all_caches()
    return outs
