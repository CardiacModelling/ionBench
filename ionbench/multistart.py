def multistart(opt, bm, initParams, filename, **kwargs):
    for i in range(len(initParams)):
        out = opt(bm, x0=initParams[i], **kwargs)
        bm.tracker.save(filename+'_run'+str(i)+'.pickle')
        print(out)
        bm.reset()
