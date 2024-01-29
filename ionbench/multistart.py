def multistart(opt, bm, initParams, filename, **kwargs):
    outs = []
    for i in range(len(initParams)):
        out = opt(bm, x0=initParams[i], **kwargs)
        if not filename == '':
            bm.tracker.save(filename + '_run' + str(i) + '.pickle')
        print(out)
        outs.append(out)
        bm.reset(fullReset = False)
    return outs
