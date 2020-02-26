from torch import optim


def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    #optimizer = optim.Adam(params, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_PAR(network, epochs, lr, par_lr, train_all, nesterov=False):
    step_size = int(epochs * .8)
    params = network.get_params()
    optimizer1 = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=step_size)
    params_par = network.get_par_params()
    optimizer_par = optim.SGD(params_par, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=par_lr)
    scheduler_par = optim.lr_scheduler.StepLR(optimizer_par, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer1, scheduler1, optimizer_par, scheduler_par