import os
import shutil
import pickle
import torch


def mkdir(path):
    if os.path.isdir(path):
        is_del = input('Delete ' + path + ' Y/N:')
        if is_del.strip().lower() == 'y':
            shutil.rmtree(path)
        else:
            exit()

    os.makedirs(path)


def load_graphs(graphs_path, graphs_indices=None):
    """
    Returns a list of graphs given graphs directory and graph indices (Optional)
    If graphs_indices are not provided all graphs will be loaded
    """

    graphs = []
    if graphs_indices is None:
        for name in os.listdir(graphs_path):
            if not name.endswith('.dat'):
                continue

            with open(graphs_path + name, 'rb') as f:
                graphs.append(pickle.load(f))
    else:
        for ind in graphs_indices:
            with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
                graphs.append(pickle.load(f))

    return graphs


def save_graphs(graphs_path, graphs):
    """
    Save networkx graphs to a directory with indexing starting from 0
    """
    for i in range(len(graphs)):
        with open(graphs_path + 'graph' + str(i) + '.dat', 'wb') as f:
            pickle.dump(graphs[i], f)


# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)


def save_model(epoch, args, model, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)

    fname = args.current_model_save_path + \
        args.fname + '_' + str(epoch) + '.dat'
    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}
    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler

    for name, d in save_items.items():
        save_dict = {}
        for key, value in d.items():
            save_dict[key] = value.state_dict()

        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg

    torch.save(checkpoint, fname)


def load_model(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            for key, value in d.items():
                value.load_state_dict(checkpoint[name][key])

        if name == 'model':
            for _, value in d.items():
                value.to(device=device)


def get_model_attribute(attribute, path, device):
    fname = path
    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]
